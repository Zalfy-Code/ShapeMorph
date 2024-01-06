# ShapeMorph: 3D Shape Completion via Blockwise Discrete Diffusion
## Introduction
This repository is the official pytorch implementation of our paper: "ShapeMorph: 3D Shape Completion via Blockwise Discrete Diffusion".

## 🚀 Training

### ❄️ Dataset Preparation
We process our dataset as [3DILG](https://github.com/1zb/3DILG).

### ⛄ Shape Encoding
```
torchrun --nproc_per_node=4 run_vqvae.py --output_dir output/vqvae --model vqvae_512 --batch_size 32 --num_workers 40 --lr 1e-3 --disable_eval --point_cloud_size 2048
```

### ☃️ Shape Completion
Please revise ```--vqvae_pth``` to checkpoint path. 
```
torchrun --nproc_per_node=4 run_diffusion.py --output_dir output/diffusion --model blockwise_diffusion --vqvae vqvae_512 --vqvae_pth output/vqvae_512 --batch_size 16 --num_workers 40 --lr 1e-3 
```

## 🛸 Sampling
Please revise ```--model_pth``` and ```--vqvae_pth``` to the checkpoints path.
```
python completion.py --model discrete_diffusion --model_pth output/diffsion --vqvae vqvae_512 --vqvae_pth output/vqvae_512 --sample_n 5 
```

## 🏁 Results
Multimodal completion on ShapeNet.
![Image](assets/ShapeNet.jpg)

Multimodal completion on PartNet.
![Image](assets/PartNet.jpg)

Multimodal completion on real-scan dataset RedWood.
![Image](assets/RedWood.jpg)

## 📬 Contact
Contact [Zalfy](zalfy_code@163.com) if you have any further questions.
