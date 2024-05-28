# FinePOSE: Fine-Grained Prompt-Driven 3D Human Pose Estimation via Diffusion Models

Created by [Jinglin Xu](http://39.108.48.32/XuWebsite/), [Yijie Guo](https://github.com/GYJHSG/), [Yuxin Peng](https://scholar.google.com/citations?user=mFsXPNYAAAAJ&hl=zh-CN)

This repository contains the PyTorch implementation for FinePOSE. (CVPR 2024, Highlight)

[[Project Page]](https://pku-icst-mipl.github.io/FinePOSE_ProjectPage/) [[arXiv]](https://arxiv.org/abs/2405.05216)

## Dependencies

Make sure you have the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* tensorboard
* CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

You should download [MATLAB](https://www.mathworks.com/products/matlab-online.html) if you want to evaluate our model on MPI-INF-3DHP dataset.

## Datasets

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M

We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).  You can download the processed data from [here](https://drive.google.com/file/d/1FMgAf_I04GlweHMfgUKzB0CMwglxuwPe/view?usp=sharing).  `data_2d_h36m_gt.npz` is the ground truth of 2D keypoints. `data_2d_h36m_cpn_ft_h36m_dbb.npz` is the 2D keypoints obatined by [CPN](https://github.com/GengDavid/pytorch-cpn).  `data_3d_h36m.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.

### MPI-INF-3DHP

We set up the MPI-INF-3DHP dataset following [P-STMO](https://github.com/paTRICK-swk/P-STMO). However, our training/testing data is different from theirs. They train and evaluate on 3D poses scaled to the height of the universal skeleton used by Human3.6M (officially called "univ_annot3"), while we use the ground truth 3D poses (officially called "annot3"). The former does not guarantee that the reprojection (used by the proposed JPMA) of the rescaled 3D poses is consistent with the 2D inputs, while the latter does. You can download our processed data from [here](https://drive.google.com/file/d/1zOM_CvLr4Ngv6Cupz1H-tt1A6bQPd_yg/view?usp=share_link). Put them in the `./data` directory. 
 

### Human3.6M

To evaluate our FinePOSE with JPMA using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/model_h36m -gpu 0,1 --nolog --evaluate best_epoch_20_10.bin -num_proposals 20 -sampling_timesteps 10 -b 4
```

### MPI-INF-3DHP
To evaluate our FinePOSE with JPMA using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/model_3dhp -gpu 0,1 --nolog --evaluate best_epoch_20_10.bin -num_proposals 20 -sampling_timesteps 10 -b 4
```
After that, the predicted 3D poses under P-Best, P-Agg, J-Best, J-Agg settings are saved as four files (`.mat`) in `./checkpoint`. To get the MPJPE, AUC, PCK metrics, you can evaluate the predictions by running a Matlab script `./3dhp_test/test_util/mpii_test_predictions_ori_py.m` (you can change 'aggregation_mode' in line 29 to get results under different settings). Then, the evaluation results are saved in `./3dhp_test/test_util/mpii_3dhp_evaluation_sequencewise_ori_{setting name}_t{iteration index}.csv`. You can manually average the three metrics in these files over six sequences to get the final results.

## Training from scratch
Trained on 2*Nvidia RTX 4090.
### Human3.6M
To train our model using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/model_h36m -gpu 0,1 --nolog
```

### MPI-INF-3DHP
To train our model using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/model_3dhp -gpu 0,1 --nolog
```




## Acknowledgement
Our code refers to the following repositories.
* [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse)
* [D3DP](https://github.com/paTRICK-swk/D3DP)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)

We thank the authors for releasing their codes.
