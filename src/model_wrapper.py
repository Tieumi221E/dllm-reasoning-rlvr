import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from diffusion_core.model import DiffusionTransformer
from typing import Optional, Tuple, List, NamedTuple, Union

class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None

class DiffusionConfig(PretrainedConfig):
    model_type = "diffusion_transformer"
    def __init__(
        self,
        vocab_size=4096,
        max_position_embeddings=2048,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=2,
        intermediate_size=3072,
        emb_dropout=0.0,
        resid_dropout=0.0,
        attention_dropout=0.0,
        tie_embeddings=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.emb_dropout = emb_dropout
        self.resid_dropout = resid_dropout
        self.attention_dropout = attention_dropout
        self.tie_embeddings = tie_embeddings

class DiffusionTransformerLM(PreTrainedModel):
    config_class = DiffusionConfig
    
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self.model = DiffusionTransformer(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            intermediate_size=config.intermediate_size,
            emb_dropout=config.emb_dropout,
            resid_dropout=config.resid_dropout,
            attention_dropout=config.attention_dropout,
            tie_embeddings=config.tie_embeddings,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> LLaDAOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        logits = self.model(input_ids, attention_mask)
        return LLaDAOutput(logits=logits)

    @property
    def device(self):
        return next(self.parameters()).device
