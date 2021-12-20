# 3D Pose Transfer with Correspondence Learning and Mesh Refinement

#### [Project](https://chaoyuesong.github.io/3d-corenet/) |   [Paper](https://openreview.net/pdf?id=fG01Z_unHC)

**3D Pose Transfer with Correspondence Learning and Mesh Refinement** <br>
Chaoyue Song,
Jiacheng Wei,
Ruibo Li,
[Fayao Liu](https://sites.google.com/site/fayaoliu/),
[Guosheng Lin](https://guosheng.github.io/) <br>
in NeurIPS, 2021.

<img src="files/3dpt.gif" width="60%" height="60%" /> <br>

## Getting Started

### Installation
- Clone this repo:
```bash
git clone https://github.com/ChaoyueSong/3d-corenet.git
cd 3d-corenet
```

- Install the dependencies. Our code is based on Python 3.6, PyTorch 1.8. And we also need pymesh.
```bash
conda install -c conda-forge pymesh2
```

- Clone the Synchronized-BatchNorm-PyTorch repo.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Dataset preparation
We use [SMPL](https://smpl.is.tue.mpg.de/) as the human mesh data, please download data [here](https://drive.google.com/drive/folders/11LbPXbDg4F_pSIr0sMHzWI08FOC8XvSY).

## Generating Meshes Using Pretrained model

Download the pretrained model from [pretrained model link](https://drive.google.com/drive/folders/1BEBBENbEr9tutZsyGGc3REUuuOYqf6M3?usp=sharing) and save them in `checkpoints/`. Then run the command 
````bash
python test.py --dataset_mode human --dataroot [Your data path] --gpu_ids 0
````
The results will be saved in `./human_mesh_test/` by default.

## Training
To train new models, please run:
```bash
python train.py --dataset_mode human --dataroot [Your data path] --niter 100 --niter_decay 100 --batchSize 8 --gpu_ids 0,1
```

## Citation
If you use this code for your research, please cite the following work.

```bash
@inproceedings{song20213d,
  title={3D Pose Transfer with Correspondence Learning and Mesh Refinement},
  author={Song, Chaoyue and Wei, Jiacheng and Li, Ruibo and Liu, Fayao and Lin, Guosheng},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

## Acknowledgments

This codebase is heavily based on [CoCosNet](https://github.com/microsoft/CoCosNet). We also use Optimal Transport code from [FLOT](https://github.com/valeoai/FLOT), Data and Edge loss code from [NPT](https://github.com/jiashunwang/Neural-Pose-Transfer).

We thank all authors for the wonderful code!


