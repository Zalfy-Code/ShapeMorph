# ShapeMorph: 3D Shape Completion via Blockwise Discrete Diffusion
## Introduction
This repository is the official pytorch implementation of our paper: "ShapeMorph: 3D Shape Completion via Blockwise Discrete Diffusion".

## 🚀 Training

### ❄️ Dataset Preparation
We process our dataset as [3DILG](https://github.com/1zb/3DILG).

### ⛄ Shape Encoding
Please command out ``` from shapenet_partial import ShapeNet ``` in ``` dataset.py ``` before training. 
```
torchrun --nproc_per_node=4 run_vqvae.py --output_dir output/vqvae --model vqvae_512 --batch_size 32 --num_workers 40 --lr 1e-3 --disable_eval --point_cloud_size 2048
```

### ☃️ Shape Completion
Please command out ``` from shapenet import ShapeNet ``` in ``` dataset.py ``` before training. 
```
torchrun --nproc_per_node=4 run_diffusion.py --output_dir output/diffusion --model blockwise_diffusion --vqvae vqvae_512 --vqvae_pth output/vqvae_512 --batch_size 16 --num_workers 40 --lr 1e-3 
```

## 🛸 Sampling
```
python completion.py --model auto_completion --model_pth output/auto --sdvq sdvq_513 --sdvq_pth output/sdvq --sample_n 5 --results results
```
