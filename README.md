# MRI reconstruction

This is the code for our recent work that develops a new method for MRI reconstruction.
The code is created and modified based upon the collaborative research project from Facebook AI Research (FAIR)
and NYU Langone Health. The code to their work is at https://github.com/facebookresearch/fastMRI.

This repository contains convenient PyTorch data loaders, subsampling functions, evaluation
metrics, and reference implementations of our methods for MRI reconstruction.


## Citing
If you use this code in your research, please consider citing
the original fastMRI dataset paper:
```
@inproceedings{zbontar2018fastMRI,
  title={fastMRI: An Open Dataset and Benchmarks for Accelerated MRI},
  author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Matthew J. Muckley and Mary Bruno and Aaron Defazio and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and James Pinkerton and Duo Wang and Nafissa Yakubova and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1811.08839},
  year={2018}
}
```
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
python models/unet/train_unet.py


## License
fastMRI is MIT licensed, as found in the LICENSE file.
