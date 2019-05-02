"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.PReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.PReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    
class ResBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob, same='False'):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_chans)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_chans)
        self.same = same
        
    def forward(self, input):
        shortcuts = self.conv(input)
        if self.same == 'True':
            shortcuts = input
        out = self.conv1(input)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += shortcuts
        out = self.relu(out)
        return out
    
    
class AttBlock(nn.Module):
    def __init__(self, in_chans, resolution, drop_prob, gate_bias=-1.0):
        super().__init__()

        self.in_chans = in_chans
        self.resolution = resolution
        self.drop_prob = drop_prob

        self.q_g = nn.Linear(in_chans//8, resolution**2)
        self.q_l_conv = nn.Conv2d(in_chans, in_chans//8, kernel_size=1)
        self.k_conv = nn.Conv2d(in_chans, in_chans//8, kernel_size=1)
        self.v_conv = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        
#         self.gate = nn.Linear(in_chans//8, resolution**2)
#         self.gate.bias.data.fill_(gate_bias)
        self.gamma_g = nn.Parameter(torch.zeros(1))
        self.gamma_l = nn.Parameter(torch.zeros(1))        
#         self.softmax = nn.Softmax(dim=-1)
        
    def init_weights(self, init_range=0.1):
        self.q_g.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, input):
        bs, c, h, w = input.size()
        q_l_flat = self.q_l_conv(input).view(bs, -1, h*w).permute(0,2,1) #bs x (h*w) x c
        k_g_flat = self.k_conv(input).view(bs, h*w, -1) # bs x (h*w) x c
        k_l_flat = self.k_conv(input).view(bs, -1, h*w) #bs x c x (h*w)
        alphas_g = self.q_g(k_g_flat) # bs x (h*w) x (h*w)
        alphas_l = torch.bmm(q_l_flat, k_l_flat) # bs x (h*w) x (h*w)
        
#         gate = F.sigmoid(self.gate(k_g_flat)) # bs x (h*w) x (h*w)
        
        alphas = self.gamma_g * alphas_g + self.gamma_l * alphas_l # element-wise multiplication

        v_flat = self.v_conv(input).view(bs, -1, h*w) #bs x c x (h*w)
        
        wed_sum = torch.bmm(v_flat, alphas.permute(0,2,1)) #bs x c x (h*w)
        out = wed_sum.view(bs, c, h, w)
        out = out + input       
        return out
        
        

class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, resolution):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.resolution = resolution

        self.down_sample_layers = nn.ModuleList([ResBlock(in_chans, chans, drop_prob)])
        ch = chans
        reso = resolution
        self.skip_layers = nn.ModuleList([AttBlock(ch, reso, drop_prob)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ResBlock(ch, ch * 2, drop_prob)]
            ch *= 2
            reso //= 2
            self.skip_layers += [AttBlock(ch, reso, drop_prob)]
            
        self.bottom_layers = nn.ModuleList([ResBlock(ch, ch, drop_prob, same='True')])
        self.bottom_layers += [AttBlock(ch, reso//2, drop_prob)]

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ResBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ResBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        for layer in self.bottom_layers:
            output = layer(output)

        # Apply up-sampling layers
        for i, layer in enumerate(self.up_sample_layers):
            if i <= 1:
                output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
                output = torch.cat([output, self.skip_layers[3-i](stack.pop())], dim=1)
                output = layer(output)
            else:
                output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
                output = torch.cat([output, stack.pop()], dim=1)
                output = layer(output)
        return self.conv2(output)+input
