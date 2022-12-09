# Unsupervised 3D Pose Transfer with Cross Consistency and Dual Reconstruction

#### [Project](https://chaoyuesong.github.io/X-DualNet/) |   [Paper](https://arxiv.org/abs/2211.10278)

**Unsupervised 3D Pose Transfer with Cross Consistency and Dual Reconstruction** <br>
Chaoyue Song,
Jiacheng Wei,
Ruibo Li,
[Fayao Liu](https://sites.google.com/site/fayaoliu/),
[Guosheng Lin](https://guosheng.github.io/) <br>
Submitted to TPAMI.

<img src="misc/network.png" width="100%" height="100%" /> <br>

The code is coming soon!

## Installation
- Clone this repo:
```bash
git clone https://github.com/ChaoyueSong/X-DualNet.git
cd X-DualNet
```

- Install the dependencies. Our code has been tested on Python 3.6, PyTorch 1.8 (previous versions also work, plz install it depends on your cuda version). We also need pymesh and open3d.
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge pymesh2
conda install -c open3d-admin open3d
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

## Citation
If you find our work is useful to your research, please consider citing the paper:

```bash
@article{song2022unsupervised,
  title={Unsupervised 3D Pose Transfer with Cross Consistency and Dual Reconstruction},
  author={Song, Chaoyue and Wei, Jiacheng and Li, Ruibo and Liu, Fayao and Lin, Guosheng},
  journal={arXiv preprint arXiv:2211.10278},
  year={2022}
}
```



