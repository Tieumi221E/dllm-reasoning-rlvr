"""
TraceRL / GRPO core for diffusion LMs.

The optimisation follows TraceRL (arXiv:2509.06949): a value-model-free, `step_map`
trajectory-decomposed PPO-clip with a k3 KL penalty. The gated reward
`R = 0.8·m_P + 0.2·m_A` (R = 0 when m_A = 0) follows the AR-baseline study (arXiv:2605.26934).

Model-agnostic: callers pass their own `mask_id`. Used by both the small-model and the
LLaDA pipelines; the small-model RL training (`src.small_train_rl`) imports
`compute_reward`, `_compute_logp_old`, `_tracerl_ppo_backward` from here.
"""
import re

import torch
import torch.nn.functional as F

from src.scoring import _normalize, _check_answer, _check_process

MASK_ID = 126336  # default only; callers always pass the model's own mask id


# ── Reward (gated R = 0.8·m_P + 0.2·m_A) ──────────────────────────────────────

def _graded_process(gen_text: str, task: dict) -> float:
    """Optional continuous process score (deductive only): matched / gold State lines.

    Off by default; the main line keeps the paper's binary m_P (comparable with 2605.26934).
    Non-deductive tasks fall back to the binary _check_process.
    """
    task_type     = task.get("task_type", "")
    gold_solution = task.get("solution", "")
    if task_type in ("deductive", "deduction_full_info", "deduction_hard"):
        gold_states = re.findall(r"State:\s*([^.]+)\.", gold_solution, re.IGNORECASE)
        if not gold_states:
            return 1.0
        gen_states = re.findall(r"State:\s*([^.]+)\.", gen_text, re.IGNORECASE)
        # position-aligned line match; extra lines are neither rewarded nor penalized (denominator = gold lines)
        n_match = sum(1 for g, p in zip(gold_states, gen_states) if _normalize(g) == _normalize(p))
        return n_match / len(gold_states)
    return float(_check_process(gen_text, task))


def compute_reward(gen_text: str, task: dict, graded_process: bool = False) -> tuple:
    """(reward, m_p, m_a).  R = 0.8·m_P + 0.2·m_A if m_A = 1, else 0.

    Evaluation (in `src.scoring`) always uses binary strict m_P ∧ m_A for comparability.
    """
    # A deductive gold answer looks like "Therefore, the answer is X."; for abductive,
    # task['answer'] is the missing-event phrase itself, with no "the answer is" prefix. Handle both.
    gold_sol = task.get("answer", "")
    m = re.search(r"Therefore, the answer is (.+?)\.", gold_sol, re.IGNORECASE)
    short_answer = m.group(1).strip() if m else gold_sol

    m_a = float(_check_answer(gen_text, short_answer, task.get("equivalent_answers")))
    if not m_a:
        return 0.0, 0.0, 0.0
    m_p = _graded_process(gen_text, task) if graded_process else float(_check_process(gen_text, task))
    return 0.8 * m_p + 0.2, m_p, m_a


# ── TraceRL PPO-clip + KL ─────────────────────────────────────────────────────

def _collapse_steps(order_full: torch.Tensor, prompt_len: int, k: int) -> torch.Tensor:
    """Merge up to k unique step-values in response positions (shrink parameter)."""
    resp  = order_full[prompt_len:]
    valid = resp[resp >= 0]
    if len(valid) == 0:
        return order_full
    unique_vals = torch.unique(valid, sorted=True)
    if len(unique_vals) <= k:
        return order_full
    n      = len(unique_vals)
    result = order_full.clone()
    for i, v in enumerate(unique_vals.tolist()):
        bucket     = min(int(i * k / n), k - 1)
        bucket_val = unique_vals[int(bucket * n / k)].item()
        result[order_full == v] = bucket_val
    return result


@torch.no_grad()
def _compute_logp_old(
    model,
    full_ids:   torch.Tensor,  # (L,) on device - prompt + generated tokens
    step_map:   torch.Tensor,  # (gen_len,) CPU
    prompt_len: int,
    mask_id:    int = MASK_ID,
    shrink:     int = 8,
    chunk:      int = 4,       # trajectory steps batched per forward
) -> list:
    """
    Pre-compute log P_old(x_t | noisy_ids_t) for every trajectory step.
    MUST be called with model.eval() and inside torch.no_grad(), BEFORE optimizer.step().
    Returns list of (pmask, mask_pos, logp_old_cpu) - one entry per shrunk step.
    """
    device     = full_ids.device
    L          = len(full_ids)
    order_full = torch.full((L,), -1, dtype=torch.long, device=device)
    order_full[prompt_len:] = step_map.to(device)
    if shrink > 0:
        order_full = _collapse_steps(order_full, prompt_len, shrink)
    uniq_steps = torch.unique(order_full[prompt_len:], sorted=True)
    uniq_steps = uniq_steps[uniq_steps >= 0]

    steps_meta = []
    for sv in uniq_steps:
        pmask    = (order_full == sv)           # target tokens at this step
        mask_pos = (order_full >= sv)           # all positions still masked
        steps_meta.append((pmask, mask_pos))

    step_list = []
    for pmask, mask_pos in steps_meta:
        # incremental canvas: the rollout sampler truncates the sequence at the
        # end of the current block group - future blocks do not exist, so the
        # reconstructed state must truncate too (never full-canvas masks).
        end = int(pmask.nonzero().max().item()) + 1
        noisy = full_ids[:end].masked_fill(pmask[:end], mask_id).unsqueeze(0)
        logits = model(noisy).logits[0]          # (end, V)
        rows     = logits[pmask[:end]].float()
        logp     = F.log_softmax(rows, dim=-1)
        logp_old = logp.gather(1, full_ids[:end][pmask[:end]].unsqueeze(1)).squeeze(1).cpu()
        step_list.append((pmask, mask_pos, logp_old))
    return step_list


def _tracerl_ppo_backward(
    model,
    full_ids:   torch.Tensor,  # (L,) on device
    step_list:  list,          # from _compute_logp_old
    advantage:  float,
    normalizer: float,         # n_active rollouts - keeps gradient scale stable
    eps:        float = 0.2,
    beta:       float = 0.01,
    use_kl_k3:  bool  = True,  # k3 unbiased KL estimator (default in TraceRL)
    mask_id:    int   = MASK_ID,
    chunk:      int   = 2,     # trajectory steps batched per forward+backward
) -> tuple:
    """
    PPO-clip + KL loss for one rollout, backward() per chunk of steps.
    Batching steps into one forward is mathematically identical to per-step
    backward (losses are summed; each step's loss only depends on its own row).
    Returns (policy_loss, kl_loss, clip_frac, ratio_mean) averaged over steps.
    """
    device  = full_ids.device
    n_steps = len(step_list)
    adv_t   = torch.tensor(advantage, dtype=torch.float32, device=device)

    tot_policy = tot_kl = tot_clip = tot_ratio = 0.0

    for i in range(0, n_steps, 1):
        sub = step_list[i:i + 1]
        chunk_loss = None
        for (pmask, mask_pos, logp_old_cpu) in sub:
            # same incremental-canvas state as _compute_logp_old
            end   = int(pmask.nonzero().max().item()) + 1
            noisy = full_ids[:end].masked_fill(pmask[:end], mask_id).unsqueeze(0)
            logits = model(noisy).logits[0]
            logp_old = logp_old_cpu.to(device)                            # (n_mask,)
            rows     = logits[pmask[:end]].float()                        # (n_mask, V)
            logp_new = F.log_softmax(rows, dim=-1).gather(
                1, full_ids[:end][pmask[:end]].unsqueeze(1)
            ).squeeze(1)                                                  # (n_mask,)

            # PPO-clip objective
            ratio   = torch.exp(logp_new - logp_old)                      # (n_mask,)
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
            surr    = torch.min(ratio * adv_t, clipped * adv_t)
            policy_loss = -surr.mean()

            # KL penalty - added to loss, not just logged
            kl = logp_new - logp_old
            kl_pen = (-kl).exp() - 1.0 + kl if use_kl_k3 else kl          # k3 or k1
            kl_loss = beta * kl_pen.mean()

            step_loss  = (policy_loss + kl_loss) / (normalizer * n_steps)
            chunk_loss = step_loss if chunk_loss is None else chunk_loss + step_loss

            with torch.no_grad():
                tot_policy += policy_loss.item()
                tot_kl     += kl_loss.item()
                tot_clip   += ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float().mean().item()
                tot_ratio  += ratio.mean().item()

        chunk_loss.backward()                     # frees this chunk's graph

    n = max(n_steps, 1)
    return tot_policy / n, tot_kl / n, tot_clip / n, tot_ratio / n
