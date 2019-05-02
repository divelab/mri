# MRI reconstruction

This is the code for our recent work that develops a new method for MRI reconstruction.
The code is created and modified based upon the collaborative research project from Facebook AI Research (FAIR)
and NYU Langone Health. The code to their work is at https://github.com/facebookresearch/fastMRI.

This repository contains convenient PyTorch data loaders, subsampling functions, evaluation
metrics, and reference implementations of our methods for MRI reconstruction.


## Citing
Citing bibtex for our work will be avialable upon the publishing of our paper.


## Dependencies
We have tested this code using:
* Ubuntu 18.04
* Python 3.6
* CUDA 9.0
* CUDNN 7.0

You can find the full list of Python packages needed to run the code in the
`requirements.txt` file. These can be installed using:
```bash
pip install -r requirements.txt
```

## Train and test
At the root directory, run
```bash
python models/unet/train_unet.py
```


## License
fastMRI is MIT licensed, as found in the LICENSE file.
