"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import random
import shutil
import time
import os
from collections import defaultdict

import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim
import h5py
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append('/mnt/dive/shared/yiliu/isl/MRI/oasis_lg/')
from common.args import Args
from common.subsample import MaskFunc
from common.utils import save_reconstructions
from data import transforms
from models.unet.unet_model import UnetModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        for fname in sorted(files):
            num_slices = 1
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            image = np.array(data['image'])
            target = np.array(data['target'])
            return self.transform(image, target, data.attrs, fname.name, slice)



class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution):

        self.resolution = resolution

    def __call__(self, image, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """

        image = transforms.to_tensor(image)

        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = transforms.to_tensor(target)
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return image, target, mean, std, fname, slice


def create_datasets(args):
    train_data = SliceData(
        root=args.data_path / 'oasis_train',
        transform=DataTransform(args.resolution),
    )
    
    dev_data = SliceData(
        root=args.data_path / 'oasis_val',
        transform=DataTransform(args.resolution),
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        input, target, mean, std, _, _ = data
        input = input.unsqueeze(1).to(args.device)
        target = target.to(args.device)

        output = model(input).squeeze(1)
        loss = F.l1_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            print('Epoch: %d / %d'%(epoch, args.num_epochs), end=',  ')
            print('Iter: %d / %d'%(iter, len(data_loader)), end=',  ')
            print('Loss: %.4g'%loss.item(), end=',  ')
            print('Avg Loss: %.4g'%avg_loss, end=',  ')
            print('Time: %.4fs'%float(time.perf_counter() - start_iter))
            

        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, _, _ = data
            input = input.unsqueeze(1).to(args.device)
            target = target.to(args.device)
            output = model(input).squeeze(1)

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            target = target * std + mean
            output = output * std + mean

            loss = F.mse_loss(output, target, size_average=False)
            losses.append(loss.item())
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, _, _= data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            output = model(input)
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=str(exp_dir / 'model.pt')
    )
    if is_new_best:
        shutil.copyfile(str(exp_dir / 'model.pt'), str(exp_dir / 'best_model.pt'))


def build_model(args):
    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        resolution=args.resolution
    ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimizer



### below are the reconstruction and evaluate_metrics parts
def load_model_metrics(checkpoint_file):
    checkpoint = torch.load(str(checkpoint_file))
    args = checkpoint['args']
    model = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob, args.resolution).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


# def run_unet(args, model, data_loader):
#     model.eval()
#     reconstructions = defaultdict(list)
#     with torch.no_grad():
#         for data in data_loader:
#             input, _, mean, std, fnames, slices = data
#             input = input.unsqueeze(1).to(args.device)
#             recons = model(input).to('cpu').squeeze(1)
#             for i in range(recons.shape[0]):
#                 recons[i] = recons[i] * std[i] + mean[i]
#                 reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

#     reconstructions = {
#         fname: np.stack([pred for _, pred in sorted(slice_preds)])
#         for fname, slice_preds in reconstructions.items()
#     }
#     return reconstructions

def run_unet(args, model, data_loader_metrics):
    model.eval()
    reconstructions = defaultdict(list)
    input_array = np.zeros((256, 256, 416), dtype=np.float64)
    target_array = np.zeros((256, 256, 416), dtype=np.float64)
    recons_array = np.zeros((256, 256, 416), dtype=np.float64)
    with torch.no_grad():
        for i, data in enumerate(data_loader_metrics):
            if i<26:
                input, target, mean, std, fnames, slices = data
            
                input = input.unsqueeze(1).to(args.device)
                recons = model(input).to('cpu').squeeze(1).permute(1,2,0)
                input_visual = input.squeeze(1)
                input_visual = input_visual.permute(1,2,0)
                input_visual = input_visual.to('cpu')
                input_visual = input_visual.numpy()
                input_array[:,:,i*len(data[0]):(i+1)*len(data[0])] = input_visual
            
                target = target.to(args.device)
                target_visual = target.permute(1,2,0)
                target_visual = target_visual.to('cpu')
                target_visual = target_visual.numpy()
                target_array[:,:,i*len(data[0]):(i+1)*len(data[0])] = target_visual
            
                recons_visual = recons.numpy()
                recons_array[:,:,i*len(data[0]):(i+1)*len(data[0])] = recons_visual
        np.save('./results_visual/input.npy', input_array)
        np.save('./results_visual/target.npy', target_array)
        np.save('./results_visual/recons.npy', recons_array)
        np.save('./results_visual/error.npy', target_array-recons_array)
        print('End the saving of all arrays')
        


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return '   '.join(
            '{name}'.format(name=name)  + ' = ' + '%.4g'%means['{name}'.format(name=name)] + ' +/- ' + '%.4g'%(2 * stddevs['{name}'.format(name=name)]) for name in metric_names
        )

def evaluate_metrics(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)
    target_path = args.data_path / 'oasis_val'
    predictions_path = args.out_dir 
    for tgt_file in target_path.iterdir():
        with h5py.File(tgt_file) as target, h5py.File(predictions_path / tgt_file.name) as recons:
            target = target['target'].value[np.newaxis, :]
            recons = recons['reconstruction'].value
            metrics.push(target, recons)
    return metrics



def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer =SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    resume_flag = 0
    if args.resume:
        resume_flag = 1
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    
    print('----------Start Training----------')
    print('resume_flag is', resume_flag)
    if resume_flag == 0:
        print('Because we run from scratch, we need to delete the previous results file')
        os.remove('./models/unet/results.txt')
    f = open('./models/unet/results.txt', 'a+')
    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        print('Epoch: %d / %d'%(epoch, args.num_epochs), end=',  ')
        print('TrainLoss: %.4g'%train_loss, end=',  ')
        print('DevLoss: %.4g'%dev_loss, end=',  ')
        print('TrainTime: %.4fs'%float(train_time), end=',  ')
        print('DevTime: %.4fs'%float(dev_time))
        
        # get metrics
        if epoch % args.metric_interval == 0:
#             data_loader_metrics = create_data_loaders_metrics(args)
#             a = 10
#             print('Now stop %ds temporarily'%a)
#             time.sleep(a)
            print('Now start reconstructing!') 
            model_metrics = load_model_metrics(args.checkpoint_metrics)
            reconstructions = run_unet(args, model_metrics, dev_loader)
            save_reconstructions(reconstructions, args.out_dir)
            recons_key = 'reconstruction_esc' if args.challenge == 'singlecoil' else 'reconstruction_rss'
            metrics = evaluate_metrics(args, recons_key)
            print(metrics)
            f.write('Epoch:{epoch}'.format(epoch=epoch)+'  '+str(metrics)+'\n')

    f.close()
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=10, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='./models/unet/checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='./models/unet/checkpoints/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    
    # get metrics
    parser.add_argument('--metric-interval', type=int, default=1, help='Period of report metrics')
    parser.add_argument('--checkpoint-metrics', type=pathlib.Path, default='./models/unet/checkpoints/model.pt', help='Which model to use, best model or recent model.')
    parser.add_argument('--out-dir', type=pathlib.Path, default='./models/unet/reconstructions_val', help='Path to save the reconstructions to')
    parser.add_argument('--acquisition', type=str, default='CORPDFS_FBK', help='PD or PDFS, if set, only volumes of the specified acquisition type are used for evaluation. By default, all volumes are included.')
    parser.add_argument('--accelerations-metrics', nargs='+', default=[4], type=int, help='If set, only volumes of the specified acceleration rate are used for evaluation. By default, all volumes are included.')
    parser.add_argument('--center-fractions-metrics', nargs='+', default=[0.08], type=float, help='fraction of low-frequency k-space columns to be sampled. Should have the same length as accelerations.')
    
    return parser


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
