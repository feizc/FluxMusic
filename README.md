## FluxMusic: Text-to-Music Generation with Rectified Flow Transformer <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv-2407.11633-b31b1b.svg)](https://arxiv.org/abs/2407.11633)

This repo contains PyTorch model definitions, pre-trained weights, and training/sampling code for paper *Flux that plays music*. 


### 1. Training 

You can refer to the [link](https://github.com/black-forest-labs/flux) to build the running environment.

To launch DiT-MoE-S/2 (256x256) in the latent space training with `N` GPUs on one node with pytorch DDP:
```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
--model DiT-S/2 \
--data-path /path/to/imagenet/train \
--image-size 256 \
--global-batch-size 256 \
--vae-path /path/to/vae
```


### 2. Inference 

We include a [`sample.py`](sample.py) script which samples images from a DiT-MoE model. Take care that we use torch.float16 for large model inference. 
```bash
python sample.py \
--model DiT-XL/2 \
--ckpt /path/to/model \
--vae-path /path/to/vae \
--image-size 256 \
--cfg-scale 1.5
```


### 3. Download Models and Data 

We are processing it as soon as possible, the model weights, data and used scripts for results reproduce will be released within two weeks continuously :) 

We use VAE in AudioLDM2, CLAP-L, and T5-XXL. 


|  Model |  Url | Scripts |  
|---------------|------------------|---------| 
| Small         | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_s_8E2A.pt)  |   | 
| Base          | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_s_16E2A.pt)  |  |  
| Large         | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_b_8E2A.pt)  |   | 
| Giant         | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_xl_8E2A.pt)   |  | 


### Acknowledgments

The codebase is based on the awesome [Flux](https://github.com/black-forest-labs/flux) and [AudioLDM2](https://github.com/haoheliu/AudioLDM2) repos. 




