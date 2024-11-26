## FluxMusic: Text-to-Music Generation with Rectified Flow Transformer <br><sub>Official PyTorch Implementation</sub>

<a href="https://arxiv.org/abs/2409.00587"><img src="https://img.shields.io/static/v1?label=Paper&message=FluxMusic&color=purple&logo=arxiv"></a> &ensp;
<a href="https://huggingface.co/feizhengcong/fluxmusic"><img src="https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://github.com/feizc/FluxMusic"><img src="https://img.shields.io/static/v1?label=Webpage&message=Cases&color=green"></a> &ensp;
  <a href="https://github.com/curtified/FluxMusicGUI"><img src="https://img.shields.io/static/v1?label=GUI&message=FluxMusic&color=orange&logo=demo"></a> &ensp;
</div>


This repo contains PyTorch model definitions, pre-trained weights, and training/sampling code for paper *Flux that plays music*. 
It explores a simple extension of diffusion-based rectified flow Transformers for text-to-music generation. The model architecture can be seen as follows: 

<img src=visuals/framework.png width=400 />


### To-do list

- [x] training / inference scripts
- [x] clean code
- [x] all ckpts and part of dataset


### 1. Training 

You can refer to the [link](https://github.com/black-forest-labs/flux) to build the running environment.

To launch small version in the latent space training with `N` GPUs on one node with pytorch DDP:
```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
--version small \
--data-path xxx \
--global_batch_size 128
```

More scripts of different model size can reference to `scripts` file direction. 


### 2. Inference 

We include a [`sample.py`](sample.py) script which samples music clips according to conditions from a MusicFlux model as:  
```bash
python sample.py \
--version small \
--ckpt_path /path/to/model \
--prompt_file config/example.txt
```

All prompts used in paper are lists in `config/example.txt`.


### 3. Download Ckpts and Data 

We use VAE and Vocoder in AudioLDM2, CLAP-L, and T5-XXL. You can download in the following table directly, we also provide the training scripts in our experiments. 

Note that in actual experiments, a restart experiment was performed due to machine malfunction, so there will be resume options in some scripts.


|  Model |Training steps  |  Url | Training scripts |  
|-------|--------|------------------|---------| 
| VAE | -| [link](https://huggingface.co/cvssp/audioldm2/tree/main/vae) | - |
| Vocoder |-| [link](https://huggingface.co/cvssp/audioldm2/tree/main/vocoder) | - |
| T5-XXL | - | [link](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main/text_encoder_3) | - |
| CLAP-L | -|  [link](https://huggingface.co/laion/larger_clap_music/tree/main) | - |
| FluxMusic-Small |   200K     | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_s.pt)  |  [link](https://github.com/feizc/FluxMusic/blob/main/scripts/train_s.sh) | 
| FluxMusic-Base   |   200K    | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_b.pt)  | [link](https://github.com/feizc/FluxMusic/blob/main/scripts/train_b.sh) |  
| FluxMusic-Large   |  200K    | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_l.pt)  | [link](https://github.com/feizc/FluxMusic/blob/main/scripts/train_l.sh)  | 
| FluxMusic-Giant    |  200K   | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_g.pt)   | [link](https://github.com/feizc/FluxMusic/blob/main/scripts/train_g.sh) | 
| FluxMusic-Giant-Full    |  2M   | [link](https://huggingface.co/feizhengcong/FluxMusic/blob/main/musicflow_g_full.pt)   | - | 


Note that 200K-steps ckpts are trained on a sub-training set and used for ploted the scaling experiments as well as case studies in the paper. 
The full version of main results will be released right way. 

The construction of training data can refer to the `test.py` file, showing a simple build of combing differnet datasets in json file. 

Considering copyright issues, the data used in the paper needs to be downloaded by oneself.  

We provide a clean subset in: <a href="https://huggingface.co/datasets/feizhengcong/FluxMusic"><img src="https://img.shields.io/static/v1?label=Datasets&message=HuggingFace&color=blue"></a> &ensp; 

A quick download link for other datasets can be found in [Huggingface](https://huggingface.co/datasets?search=music) : ). 

This is a research project, and it is recommended to try advanced products: 
<a href="https://www.melodio.ai/"><img src="https://img.shields.io/static/v1?label=Recommend&message=Application&color=orange&logo=demo"></a> &ensp; 


### Acknowledgments

The codebase is based on the awesome [Flux](https://github.com/black-forest-labs/flux) and [AudioLDM2](https://github.com/haoheliu/AudioLDM2) repos. 




