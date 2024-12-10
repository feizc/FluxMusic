import torch 
import os
import argparse 
import logging
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from glob import glob
import yaml 
from collections import OrderedDict 
from time import time 
from einops import rearrange, repeat

from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
from audioldm2.utilities.data.dataset import AudioDataset

from constants import build_model
from utils import load_clip, load_clap, load_t5
from thop import profile


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class RF(torch.nn.Module):
    def __init__(self, ln=True):
        super().__init__()
        self.ln = ln
        self.stratified = False

    def forward(self, model, x, **kwargs):

        b = x.size(0)
        if self.ln:
            if self.stratified:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, b + 1).to(x.device)
                z = quantiles[:-1] + torch.rand((b,)).to(x.device) / b
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                t = torch.sigmoid(z)
            else:
                nt = torch.randn((b,)).to(x.device)
                t = torch.sigmoid(nt)
        else: 
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        
        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        vtheta = model(x=zt, t=t, **kwargs) 
        # print(z1.size(), x.size(), vtheta.size())
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), {"batchwise_loss": ttloss}

    @torch.no_grad()
    def sample(self, model, z, conds, null_cond=None, sample_steps=50, cfg=2.0, **kwargs):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = model(x=z, t=t, **conds)
            if null_cond is not None:
                vu = model(x=z, t=t, **null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

    @torch.no_grad()
    def sample_with_xps(self, model, z, conds, null_cond=None, sample_steps=50, cfg=2.0, **kwargs):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            
            # print(z.size(), t.size())
            vc = model(x=z, t=t, **conds)
            if null_cond is not None:
                vu = model(x=z, t=t, **null_cond)
                vc = vu + cfg * (vc - vu)
            x = z - i * dt * vc
            z = z - dt * vc
            images.append(x)
        return images


def prepare_model_inputs(args, batch, device, vae, clip, t5,):
    text_embedding, text_embedding_mask = batch['text_embedding'], batch['text_embedding_mask']
    text_embedding_t5, text_embedding_mask_t5 = batch['text_embedding_t5'], batch['text_embedding_mask_t5']
    # print(image.size(), text_embedding.size(), text_embedding_t5.size())

    # clip & mT5 text embedding
    text_embedding = text_embedding.to(device)
    text_embedding_mask = text_embedding_mask.to(device) 
    with torch.no_grad():
        encoder_hidden_states = clip.hf_module(
            text_embedding.to(device),
            attention_mask=text_embedding_mask,
            output_hidden_states=False,
        )["pooler_output"] # ()
    
    # print(encoder_hidden_states.size())

    text_embedding_t5 = text_embedding_t5.to(device).squeeze(1)
    text_embedding_mask_t5 = text_embedding_mask_t5.to(device).squeeze(1)
    with torch.no_grad():
        output_t5 = t5.hf_module(
            input_ids=text_embedding_t5,
            attention_mask=text_embedding_mask_t5,
            output_hidden_states=False,
        )
        encoder_hidden_states_t5 = output_t5["last_hidden_state"].detach() 

    with torch.no_grad():
        image = vae.encode(batch['log_mel_spec'].unsqueeze(1).to(device)).latent_dist.sample().mul_(vae.config.scaling_factor) 

    # positional embedding
    bs, c, h, w = image.shape
    image = rearrange(image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).float()
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt_ids = torch.zeros(bs, encoder_hidden_states_t5.shape[1], 3)
    # Model conditions
    model_kwargs = dict(
        img_ids=img_ids.to(image.device),
        txt = encoder_hidden_states_t5.to(image.device).float(),
        txt_ids = txt_ids.to(image.device),
        y = encoder_hidden_states.to(image.device).float(),
    )

    return image, model_kwargs



def main(args): 
    assert torch.cuda.is_available(), "Training currently requires at least one GPU." 
    
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.version.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)


    model = build_model(args.version).to(device) 
    parameters_sum = sum(x.numel() for x in model.parameters())
    logger.info(f"{parameters_sum / 1000000.0} M")  

    if args.resume is not None: 
        print('load from: ', args.resume) 
        resume_ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)['ema'] 
        model.load_state_dict(resume_ckpt) 

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank]) 

    diffusion = RF() 
    model_path = '/maindata/data/shared/public/zhengcong.fei/dataset/dataset_music/audioldm2' 
    vae = AutoencoderKL.from_pretrained(os.path.join(model_path, 'vae')).to(device)
    # vocoder = SpeechT5HifiGan.from_pretrained(os.path.join(model_path, 'vocoder')).to(device) 
    t5 = load_t5(device, max_length=256)
    clap = load_clap(device, max_length=256)
    # clip = load_clip(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0)


    config = yaml.load(
        open(
            'config/16k_64.yaml',
            'r'
        ),
        Loader=yaml.FullLoader,
    )
    dataset = AudioDataset(
        config=config, split="train", 
        waveform_only=False, 
        dataset_json_path=args.data_path, 
        tokenizer=clap.tokenizer, 
        uncond_pro=0.1,
        text_ctx_len=77,
        tokenizer_t5=t5.tokenizer,
        text_ctx_len_t5=256,
        uncond_pro_t5=0.1, 
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,}") 
    
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time() 
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        data_iter_step = 0 
        for batch in loader: 
            latents, model_kwargs = prepare_model_inputs(args, batch, device, vae, clap, t5,) 
            loss, _ = diffusion.forward(model=model, x=latents, **model_kwargs) 
            # bug fix
            loss = loss / args.accum_iter 
            loss.backward()
            if (data_iter_step + 1) % args.accum_iter == 0:
                opt.step() 
                opt.zero_grad()
                update_ema(ema, model.module)

            data_iter_step += 1 
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0: 
                if rank == 0:
                    checkpoint = {
                        # "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    try: 
                        torch.save(checkpoint, checkpoint_path)
                    except Exception as e: 
                        print(e)
                    
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    # model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='fma_dataset.json')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--version", type=str, default="large")
    parser.add_argument("--vae-path", type=str, default='audioldm2/vae')
    parser.add_argument("--epochs", type=int, default=1400) 
    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=1234) 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument('--accum_iter', default=16, type=int,)  
    parser.add_argument("--ckpt-every", type=int, default=100_000) 
    parser.add_argument('--local-rank', type=int, default=-1, help='local rank passed from distributed launcher') 
    args = parser.parse_args() 
    main(args) 
