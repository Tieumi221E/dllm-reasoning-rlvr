import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List


KVCache = Tuple[torch.Tensor, torch.Tensor]   # (K, V) both [B, kv_heads, L, head_dim]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class GQAAttention(nn.Module):
    """Grouped Query Attention — 12 Q heads / 2 KV heads, no projection bias."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.attn_drop_p = attention_dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out, _ = self._forward_inner(x, attention_mask, past_kv=None)
        return out

    def forward_with_past(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        """Forward with optional KV cache for prefix.

        x         : [B, block_len, hidden_size]  current block embeddings only
        past_kv   : (K, V) each [B, num_kv_heads, prefix_len, head_dim], or None
        Returns (output, new_kv) where new_kv covers only the current block.
        """
        out, new_kv = self._forward_inner(x, attention_mask=None, past_kv=past_kv)
        return out, new_kv

    def _forward_inner(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_kv: Optional[KVCache],
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        new_kv: Optional[KVCache] = (k, v)  # raw (pre-expansion) for caching

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Build attention mask only when there is padding (inference without past_kv).
        if past_kv is None and attention_mask is not None and not attention_mask.all():
            pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_mask = torch.zeros(
                B, 1, 1, attention_mask.shape[1],
                dtype=x.dtype, device=x.device,
            ).masked_fill(pad, float('-inf'))
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, L, -1)), new_kv


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(self.w1(x) * F.silu(self.w2(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        attention_dropout: float,
        resid_dropout: float,
    ) -> None:
        super().__init__()
        self.attn = GQAAttention(hidden_size, num_heads, num_kv_heads, attention_dropout)
        self.attn_dropout = nn.Dropout(resid_dropout)
        self.ff = SwiGLUFeedForward(hidden_size, intermediate_size, resid_dropout)
        self.ff_dropout = nn.Dropout(resid_dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_dropout(self.attn(self.norm1(x), attention_mask))
        x = x + self.ff_dropout(self.ff(self.norm2(x)))
        return x

    def forward_with_past(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        attn_out, new_kv = self.attn.forward_with_past(self.norm1(x), past_kv=past_kv)
        x = x + self.attn_dropout(attn_out)
        x = x + self.ff_dropout(self.ff(self.norm2(x)))
        return x, new_kv


class DiffusionTransformer(nn.Module):
    """Bidirectional Transformer for Masked Diffusion (GQA, no causal mask)."""

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int = 2,
        intermediate_size: int = 3072,
        emb_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_position_embeddings, hidden_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_heads, num_kv_heads,
                intermediate_size, attention_dropout, resid_dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.token_emb.weight
        self.max_pos = max_position_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).clamp(max=self.max_pos - 1)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            if self.training:
                x = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        return self.head(self.norm(x))

    def build_kv_cache(
        self, input_ids: torch.Tensor,
    ) -> List[KVCache]:
        """Run a forward pass on prefix tokens and return per-layer KV caches.

        Used before block-by-block denoising to cache prompt representations.
        input_ids : [B, prefix_len]
        Returns   : list of (K, V) per layer, each [B, num_kv_heads, prefix_len, head_dim]
        """
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).clamp(max=self.max_pos - 1)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(pos))
        past_kvs: List[KVCache] = []
        with torch.no_grad():
            for layer in self.layers:
                x, kv = layer.forward_with_past(x, past_kv=None)
                past_kvs.append(kv)
        return past_kvs

    def forward_block(
        self,
        block_ids: torch.Tensor,
        past_kvs: List[KVCache],
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        """Forward pass for one block given prefix KV cache.

        block_ids : [B, block_len]
        past_kvs  : per-layer KV caches for prefix (from build_kv_cache or extend_kv_cache)
        Returns   : (logits [B, block_len, V], new_kvs for block only)
        """
        b, s = block_ids.shape
        prefix_len = past_kvs[0][0].shape[2]
        pos = torch.arange(
            prefix_len, prefix_len + s, device=block_ids.device
        ).unsqueeze(0).clamp(max=self.max_pos - 1)
        x = self.emb_dropout(self.token_emb(block_ids) + self.pos_emb(pos))
        new_kvs: List[KVCache] = []
        for layer, pkv in zip(self.layers, past_kvs):
            x, new_kv = layer.forward_with_past(x, past_kv=pkv)
            new_kvs.append(new_kv)
        return self.head(self.norm(x)), new_kvs

    @staticmethod
    def extend_kv_cache(
        past_kvs: List[KVCache],
        block_kvs: List[KVCache],
    ) -> List[KVCache]:
        """Append a committed block's KV to the prefix cache."""
        return [
            (torch.cat([pk, bk], dim=2), torch.cat([pv, bv], dim=2))
            for (pk, pv), (bk, bv) in zip(past_kvs, block_kvs)
        ]
