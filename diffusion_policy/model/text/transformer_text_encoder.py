import copy

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules

from transformers import CLIPTokenizer, CLIPModel

logger = logging.getLogger(__name__)

class TransformerTextEncoder(ModuleAttrMixin):
    def __init__(self,
        model_name: str="openai/clip-vit-base-patch16",
        n_emb: int=768,
        pretrained: bool=True,
        frozen: bool=True,
        default_text: str=None,
        ) -> None:
        """
        Assumes text input: B x 1
        """
        super().__init__()

        assert pretrained, "[TextEncoder] Only pretrained models are supported"
        self.model = CLIPModel.from_pretrained(model_name, )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, )
        self.model_name = model_name

        self.frozen = frozen
        if frozen:
            assert pretrained
            for param in self.model.parameters():
                param.requires_grad = False
        
        with torch.no_grad():
            eample_text = ["powerlifting is cool"]
            exmple_input = self.tokenizer(eample_text, return_tensors="pt", padding=True, truncation=False)
            example_feature = self.model.get_text_features(**exmple_input)
            feature_size = example_feature.shape[-1]

        self.out_projection = nn.Linear(in_features=feature_size, out_features=n_emb)
        self.default_text = default_text
        if default_text is not None:
            inputs = self.tokenizer([default_text], return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            self.text_embedding_before_projection = outputs
        logger.info(
                "number of parameters: %e", sum(p.numel() for p in self.parameters())
            )
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text: list):
        if np.all([t == self.default_text for t in text]):
            text_embeding = self.out_projection(self.text_embedding_before_projection.to(self.device))
            text_embeding = text_embeding.repeat(len(text), 1)
            return text_embeding
        else:
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            text_embeding = self.out_projection(outputs)
            return text_embeding