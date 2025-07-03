# Adapting DiffusionMBIR to MRI Reconstruction

## IXI Dataset

- Visualize a Slice: You can visualize a specific slice from a T1 volume and its corresponding mask using:
  - `--plane` can be `axial`, `coronal`, or `sagittal`.
  - `--slice_index` specifies which slice to visualize.

  ```bash
  python mri_utils.py \
    --mode slice \
    --t1_file $T1_FILE_PATH$ \
    --mask_file $MASK_FILE_PATH$ \
    --plane axial \
    --slice_index 70
  ```

- 3D Anatomical View in FSLeyes: To convert a T1 volume and mask to NIfTI format and open them in FSLeyes for 3D inspection:

  ```bash
  python mri_utils.py \
    --mode fsleyes \
    --t1_file $T1_FILE_PATH$ \
    --mask_file $MASK_FILE_PATH$
  ```

- Visualize a Preprocessed Slice: To visualize a slice, plot its histogram, and see the effect of mapping the top 2% of intensities to the 98th percentile:

  ```bash
  python mri_utils.py \
    --mode slice_hist \
    --t1_file $T1_FILE_PATH$ \
    --plane axial \
    --slice_index 70
  ```

Replace `$T1_FILE_PATH$` and `$MASK_FILE_PATH$` with the actual paths to your `.npy` files.

### Dataset Preprocessing 
```bash
python python mri_preprocess_IXI_slices_128.py
```

## Training
Train the diffusion model with preprocessed IXI dataset by using e.g.
```bash
bash train_IXI.sh
```
You can modify the training config with the ```--config``` flag.

# Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models (CVPR 2023)

Official PyTorch implementation of **DiffusionMBIR**, the CVPR 2023 paper "[Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models](https://arxiv.org/abs/2211.10655)". Code modified from [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).

[![arXiv](https://img.shields.io/badge/arXiv-2211.10655-green)](https://arxiv.org/abs/2211.10655)
[![arXiv](https://img.shields.io/badge/paper-CVPR2023-blue)](https://arxiv.org/abs/2211.10655)
![concept](./figs/forward_model.jpg)
![concept](./figs/cover_result.jpg)

## Getting started

### Download pre-trained model weights

* Make a conda environment and install dependencies
```bash
conda env create --file environment.yml
```

## DiffusionMBIR (fast) reconstruction
Once you have the pre-trained weights and the test data set up properly, you may run the following scripts. Modify the parameters in the python scripts directly to change experimental settings.

```bash
conda activate diffusion-mbir
python inverse_problem_solver_AAPM_3d_total.py
python inverse_problem_solver_BRATS_MRI_3d_total.py
```

## Citation
This repo is based on the following work:

```
@InProceedings{chung2023solving,
  title={Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models},
  author={Chung, Hyungjin and Ryu, Dohoon and McCann, Michael T and Klasky, Marc L and Ye, Jong Chul},
  journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
