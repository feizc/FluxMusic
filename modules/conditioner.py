from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer, AutoTokenizer, ClapTextModel)


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_t5 = version.startswith("google")
        self.max_length = max_length
        self.output_key = "last_hidden_state" if self.is_t5 else "pooler_output"

        if version.startswith("openai"): 
            local_path = 'ckpt/stable-diffusion-3-medium-diffusers'
            local_path_tokenizer = 'ckpt/stable-diffusion-3-medium-diffusers'
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(local_path_tokenizer, subfolder="tokenizer", max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(local_path, subfolder="text_encoder", **hf_kwargs).half()
        elif version.startswith("laion"): 
            local_path = "laion/clap-htsat-fused"
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, max_length=max_length)
            self.hf_module: ClapTextModel = ClapTextModel.from_pretrained(local_path, **hf_kwargs).half()
        else: 
            local_path = 'ckpt/stable-diffusion-3-medium-diffusers' 
            local_path_tokenizer = 'ckpt/stable-diffusion-3-medium-diffusers' 
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(local_path_tokenizer, subfolder="tokenizer_3", max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(local_path, subfolder="text_encoder_3", **hf_kwargs).half()

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
