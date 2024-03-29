# Unsupervised 3D Pose Transfer with Cross Consistency and Dual Reconstruction

#### [Project](https://chaoyuesong.github.io/X-DualNet/) |   [Paper](https://arxiv.org/abs/2211.10278)

**Unsupervised 3D Pose Transfer with Cross Consistency and Dual Reconstruction** <br>
Chaoyue Song,
Jiacheng Wei,
Ruibo Li,
[Fayao Liu](https://sites.google.com/site/fayaoliu/),
[Guosheng Lin](https://guosheng.github.io/) <br>
in TPAMI 2023.

<img src="misc/network.png" width="100%" height="100%" /> <br>

## Installation
- Clone this repo:
```bash
git clone https://github.com/ChaoyueSong/X-DualNet.git
cd X-DualNet
```

- Install the dependencies. Our code has been tested on Python 3.6, PyTorch 1.8 (previous versions also work, plz install it according to your cuda version). We also need pymesh and open3d.
```bash
conda env create -f environment.yml
conda activate x_dualnet
```

- Clone the Synchronized-BatchNorm-PyTorch repo.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Dataset preparation
We use [SMPL](https://smpl.is.tue.mpg.de/) as the human mesh data, please download data [here](https://drive.google.com/drive/folders/11LbPXbDg4F_pSIr0sMHzWI08FOC8XvSY). And we generate our animal mesh data using [SMAL](https://smal.is.tue.mpg.de/), please download it [here](https://drive.google.com/drive/folders/1uP6H0j7mUJ6utgvXxpT-2rn4EYhJ3el5?usp=sharing).

## Generating Meshes Using Pretrained model
By default, we load the latest checkpoint for test, which can be changed using `--which_epoch`.

#### 1) SMPL (human) 
Download the pretrained model from [pretrained model link](https://www.dropbox.com/scl/fo/f1ra0kujumdprzrm75y86/h?dl=0&rlkey=cy775d80qhcliz0aaqtxf70ab) and save them in `checkpoints/human`. Then run the command  
````bash
python test.py --dataset_mode human --dataroot [Your data path] --gpu_ids 0
````
The results will be saved in `test_results/human/`. `human_test_list` is randomly choosed for test and different from it in [3D-CoreNet](https://github.com/ChaoyueSong/3d-corenet).

#### 2) SMAL (animal) 
Download the pretrained model from [pretrained model link](https://www.dropbox.com/scl/fo/yaws6uqnvuq5pi1cms5yl/h?dl=0&rlkey=0o7oi3dtqe987wff7td7ll0vz) and save them in `checkpoints/animal`. Then run the command 
````bash
python test.py --dataset_mode animal --dataroot [Your data path] --gpu_ids 0
````
The results will be saved in `test_results/animal/`. `animal_test_list` is randomly choosed for test. For the calculation of CD and EMD, please check [TMNet](https://github.com/jnypan/TMNet) and [MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion).

## Training
#### 1) SMPL (human) 
To train new models on human meshes, please run:
```bash
python train.py --dataset_mode human --dataroot [Your data path] --niter 100 --niter_decay 100 --batchSize 4 --gpu_ids 0,1
```
The output meshes in the training process will be saved in `output/human/`.
#### 2) SMAL (animal) 
To train new models on animal meshes, please run:
```bash
python train.py --dataset_mode animal --dataroot [Your data path] --niter 100 --niter_decay 100 --batchSize 6 --gpu_ids 0,1
```
The output meshes in the training process will be saved in `output/animal/`.


Please change the batch size and gpu_ids as you desired.

If you need continue training from checkpoint, use `--continue_train`.

## Citation
If you find our work is useful to your research, please consider citing the paper:

```bash
@ARTICLE{10076900,
  author={Song, Chaoyue and Wei, Jiacheng and Li, Ruibo and Liu, Fayao and Lin, Guosheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Unsupervised 3D Pose Transfer With Cross Consistency and Dual Reconstruction}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TPAMI.2023.3259059}}
```

## Acknowledgments

This code is heavily based on [CoCosNet](https://github.com/microsoft/CoCosNet). We rewrite the [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) architecture to ver2ver. We also use Optimal Transport and PointNet++-like convolution code from [FLOT](https://github.com/valeoai/FLOT), Data and Edge loss code from [NPT](https://github.com/jiashunwang/Neural-Pose-Transfer), and [Synchronized Batch Normalization](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).

We thank all authors for the wonderful code!


