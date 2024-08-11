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
            local_path = '/maindata/data/shared/multimodal/public/ckpts/stable-diffusion-3-medium-diffusers/text_encoder' 
            local_path_tokenizer = '/maindata/data/shared/multimodal/public/ckpts/stable-diffusion-3-medium-diffusers/tokenizer' 
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(local_path_tokenizer, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(local_path, **hf_kwargs).half()
        elif version.startswith("laion"): 
            local_path = '/maindata/data/shared/multimodal/public/dataset_music/clap'
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, max_length=max_length)
            self.hf_module: ClapTextModel = ClapTextModel.from_pretrained(local_path, **hf_kwargs).half()
        else: 
            local_path = '/maindata/data/shared/multimodal/public/ckpts/stable-diffusion-3-medium-diffusers/text_encoder_3' 
            local_path_tokenizer = '/maindata/data/shared/multimodal/public/ckpts/stable-diffusion-3-medium-diffusers/tokenizer_3' 
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(local_path_tokenizer, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(local_path, **hf_kwargs).half()

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
