import torch 
from modules.autoencoder import AutoEncoder, AutoEncoderParams
from modules.conditioner import HFEmbedder
from safetensors.torch import load_file as load_sft 


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_clap(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    return HFEmbedder("laion/larger_clap_music", max_length=256, torch_dtype=torch.bfloat16).to(device)

def load_ae(ckpt_path, device: str | torch.device = "cuda",) -> AutoEncoder:
    ae_params=AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    # Loading the autoencoder
    ae = AutoEncoder(ae_params)
    sd = load_sft(ckpt_path,)
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    ae.to(device)
    return ae
