import torch
from typing import List, Optional, Tuple, Any


class DiffusionSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        mask_token_id: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id

    # ── single-sample interface (kept for compatibility) ──────────────────

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 1024,
        steps: int = 64,
        remask_mode: str = "low_confidence_static",
        temperature: Optional[float] = None,
        num_samples: int = 1,
    ) -> List[int]:
        temp = temperature if temperature is not None else self.temperature
        self.model.eval()
        response_tokens, _ = self._denoise_block(
            prompt_ids, max_new_tokens, steps, remask_mode, temp, batch_size=1
        )
        return response_tokens[0]

    # ── batch interfaces ──────────────────────────────────────────────────

    def generate_batch(
        self,
        prompt_ids: List[int],
        num_samples: int,
        max_new_tokens: int = 1024,
        steps: int = 64,
        remask_mode: str = "low_confidence_static",
        temperature: Optional[float] = None,
    ) -> List[List[int]]:
        """Full-sequence denoising (no blocks, no KV cache)."""
        temp = temperature if temperature is not None else self.temperature
        self.model.eval()
        all_tokens, _ = self._denoise_block(
            prompt_ids, max_new_tokens, steps, remask_mode, temp, batch_size=num_samples
        )
        return all_tokens

    def generate_batch_blocked(
        self,
        prompt_ids: List[int],
        num_samples: int,
        max_new_tokens: int = 1024,
        block_size: int = 256,
        steps_per_block: int = 16,
        remask_mode: str = "low_confidence_static",
        temperature: Optional[float] = None,
    ) -> List[List[int]]:
        """Block-by-block denoising with KV cache for committed prefix.

        Speedup vs generate_batch:
          - attention over current block only: O(block²) instead of O(total²)
          - KV cache for prompt + committed blocks: computed once, reused
          - fewer steps per block (simpler local problem)

        block_size      : tokens per block (256 recommended)
        steps_per_block : denoising steps per block (16 recommended)
        """
        temp = temperature if temperature is not None else self.temperature
        self.model.eval()
        B = num_samples

        suppress = [self.mask_token_id]
        if self.pad_token_id is not None and self.pad_token_id != self.eos_token_id:
            suppress.append(self.pad_token_id)
        suppress_t = torch.tensor(suppress, dtype=torch.long, device=self.device)

        # ── build prefix KV cache (prompt, shared across all blocks) ──────
        prompt_t = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.device
        ).expand(B, -1)  # [B, prompt_len]

        with torch.no_grad():
            past_kvs = self.model.build_kv_cache(prompt_t)

        # ── block-by-block denoising ───────────────────────────────────────
        n_blocks = (max_new_tokens + block_size - 1) // block_size
        committed_blocks: List[torch.Tensor] = []
        # eos_hit[i]: sample i has already generated EOS in a previous block
        eos_hit = torch.zeros(B, dtype=torch.bool, device=self.device)
        # per-sample KV caches: start identical, diverge as samples finish early
        # We keep one shared past_kvs and pad finished samples with EOS tokens
        # so their KV cache stays valid but they don't affect active samples.

        for block_i in range(n_blocks):
            if eos_hit.all():
                break
            actual = min(block_size, max_new_tokens - block_i * block_size)
            active = ~eos_hit  # [B] samples still needing generation

            # Finished samples get a block of EOS tokens (cheap, no denoising)
            block = torch.full((B, actual), self.eos_token_id, dtype=torch.long, device=self.device)

            if active.any():
                # Only denoise the active samples
                act_idx = active.nonzero(as_tuple=True)[0]  # indices of active samples
                n_act = act_idx.shape[0]

                blk = torch.full((n_act, actual), self.mask_token_id, dtype=torch.long, device=self.device)
                conf = torch.zeros(n_act, actual, device=self.device)
                frozen_conf: Optional[torch.Tensor] = None
                initial_conf_set = torch.zeros(n_act, dtype=torch.bool, device=self.device)

                # Slice KV cache to active samples only
                past_kvs_act = [(k[act_idx], v[act_idx]) for k, v in past_kvs]

                for step_idx in range(steps_per_block):
                    with torch.no_grad():
                        logits, _ = self.model.forward_block(blk, past_kvs_act)

                    masked = (blk == self.mask_token_id)
                    if not masked.any():
                        break

                    logits[:, :, suppress_t] = -float('inf')
                    scaled = logits / max(temp, 1e-5)
                    probs = torch.softmax(scaled, dim=-1)

                    probs_flat = probs.view(n_act * actual, -1)
                    sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(1)
                    sampled = sampled_flat.view(n_act, actual)
                    c = probs.gather(2, sampled.unsqueeze(2)).squeeze(2)

                    blk = torch.where(masked, sampled, blk)
                    conf = torch.where(masked, c, conf)

                    if remask_mode == "low_confidence_static":
                        newly_done = (~(blk == self.mask_token_id).any(dim=1)) & ~initial_conf_set
                        if newly_done.any():
                            if frozen_conf is None:
                                frozen_conf = conf.clone()
                            else:
                                frozen_conf[newly_done] = conf[newly_done]
                            initial_conf_set |= newly_done

                    if step_idx < steps_per_block - 1:
                        remask_prob = (steps_per_block - step_idx - 1) / steps_per_block
                        num_remask = max(1, int(round(remask_prob * actual)))
                        conf_for_sort = (frozen_conf if (remask_mode == "low_confidence_static"
                                                         and frozen_conf is not None) else conf)
                        if remask_mode in ("low_confidence", "low_confidence_static"):
                            _, remask_idx = conf_for_sort.topk(num_remask, dim=1, largest=False)
                            blk.scatter_(1, remask_idx, self.mask_token_id)
                        else:
                            rand_mask = torch.rand(n_act, actual, device=self.device) < remask_prob
                            blk = torch.where(rand_mask,
                                              torch.full_like(blk, self.mask_token_id), blk)

                block[act_idx] = blk  # write active samples' results back

            # commit: extend KV cache for ALL samples (active used real tokens, finished used EOS)
            with torch.no_grad():
                _, block_kvs = self.model.forward_block(block, past_kvs)
            past_kvs = self.model.extend_kv_cache(past_kvs, block_kvs)

            committed_blocks.append(block)

            # update eos_hit: sample done if EOS appears anywhere in this block
            for b_idx in range(B):
                if not eos_hit[b_idx] and self.eos_token_id in block[b_idx].tolist():
                    eos_hit[b_idx] = True

        # concatenate all committed blocks and EOS-truncate
        if not committed_blocks:
            return [[] for _ in range(B)]
        full = torch.cat(committed_blocks, dim=1)  # [B, max_new_tokens]
        results = []
        for seq in full.tolist():
            if self.eos_token_id in seq:
                seq = seq[:seq.index(self.eos_token_id)]
            results.append(seq)
        return results

    # ── full-sequence denoising (original, unchanged) ─────────────────────

    def _denoise_block(
        self,
        prompt_ids: List[int],
        slots: int,
        steps: int,
        remask_mode: str,
        temperature: float,
        batch_size: int = 1,
    ) -> Tuple[List[List[int]], bool]:
        B = batch_size
        prompt_len = len(prompt_ids)

        suppress = [self.mask_token_id]
        if self.pad_token_id is not None and self.pad_token_id != self.eos_token_id:
            suppress.append(self.pad_token_id)
        suppress_t = torch.tensor(suppress, dtype=torch.long, device=self.device)

        block = torch.full((B, slots), self.mask_token_id, dtype=torch.long, device=self.device)
        confidence = torch.zeros(B, slots, device=self.device)
        frozen_confidence = None
        initial_confidence_set = torch.zeros(B, dtype=torch.bool, device=self.device)

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device).expand(B, -1)

        for step_idx in range(steps):
            full_seq = torch.cat([prompt_t, block], dim=1)
            att_mask = torch.ones_like(full_seq)

            with torch.no_grad():
                out = self.model(full_seq, att_mask)
                logits = out.logits if hasattr(out, 'logits') else out
                resp_logits = logits[:, prompt_len: prompt_len + slots, :].clone()

            masked = (block == self.mask_token_id)
            if not masked.any():
                break

            resp_logits[:, :, suppress_t] = -float('inf')
            scaled = resp_logits / max(temperature, 1e-5)
            probs = torch.softmax(scaled, dim=-1)

            probs_flat = probs.view(B * slots, -1)
            sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(1)
            sampled = sampled_flat.view(B, slots)
            conf = probs.gather(2, sampled.unsqueeze(2)).squeeze(2)

            block = torch.where(masked, sampled, block)
            confidence = torch.where(masked, conf, confidence)

            if remask_mode == "low_confidence_static":
                newly_done = (~(block == self.mask_token_id).any(dim=1)) & ~initial_confidence_set
                if newly_done.any():
                    if frozen_confidence is None:
                        frozen_confidence = confidence.clone()
                    else:
                        frozen_confidence[newly_done] = confidence[newly_done]
                    initial_confidence_set |= newly_done

            if step_idx < steps - 1:
                remask_prob = (steps - step_idx - 1) / steps
                num_remask = max(1, int(round(remask_prob * slots)))
                conf_for_sort = (frozen_confidence if (remask_mode == "low_confidence_static"
                                                       and frozen_confidence is not None)
                                 else confidence)
                if remask_mode in ("low_confidence", "low_confidence_static"):
                    _, remask_idx = conf_for_sort.topk(num_remask, dim=1, largest=False)
                    block.scatter_(1, remask_idx, self.mask_token_id)
                else:
                    rand_mask = torch.rand(B, slots, device=self.device) < remask_prob
                    block = torch.where(rand_mask,
                                        torch.full_like(block, self.mask_token_id), block)

        block_list = block.tolist()
        results = []
        any_eos = False
        for seq in block_list:
            if self.eos_token_id in seq:
                seq = seq[:seq.index(self.eos_token_id)]
                any_eos = True
            results.append(seq)

        return results, any_eos
