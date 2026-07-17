"""Bridge from the project's rollout/eval call sites to dllm samplers."""

import torch

from dllm import BlockwiseConfig, generate_blockwise


def generate_blocked(model, prompt_ids, num_samples, gen_length, block_length,
                     steps_per_block, temperature, eos_token_id, mask_token_id,
                     device):
    """Incremental block generation; returns EOS-stripped token-id lists."""
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cfg = BlockwiseConfig(
        gen_length=gen_length, block_length=block_length,
        steps_per_block=steps_per_block, temperature=temperature,
        sampling="gumbel", commit="transfer", eos_token_id=eos_token_id)
    out = generate_blockwise(model, prompt, mask_token_id, cfg,
                             num_samples=num_samples)
    return out.sequences
