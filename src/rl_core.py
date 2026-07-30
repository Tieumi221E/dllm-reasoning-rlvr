"""
TraceRL / GRPO core for diffusion LMs.

The optimisation follows TraceRL (arXiv:2509.06949): a value-model-free, `step_map`
trajectory-decomposed PPO-clip with a k3 KL penalty. The gated reward
`R = 0.8·m_P + 0.2·m_A` (R = 0 when m_A = 0) follows the AR-baseline study (arXiv:2605.26934).

Model-agnostic: callers pass their own `mask_id`. Used by both the small-model and the
LLaDA pipelines; the small-model RL training (`src.small_train_rl`) imports
`compute_reward`, `_compute_logp_old`, `_tracerl_ppo_backward` from here.
"""

import math
import re

import torch
from dllm import (
    ppo_clip_objective,
    score_trajectory_states,
    trajectory_states,
)

from src.scoring import _normalize, _check_answer, _check_process

MASK_ID = 126336  # default only; callers always pass the model's own mask id


# ── Reward (gated R = 0.8·m_P + 0.2·m_A) ──────────────────────────────────────


def _graded_process(gen_text: str, task: dict) -> float:
    """Optional continuous process score (deductive only): matched / gold State lines.

    Off by default; the main line keeps the paper's binary m_P (comparable with 2605.26934).
    Non-deductive tasks fall back to the binary _check_process.
    """
    task_type = task.get("task_type", "")
    gold_solution = task.get("solution", "")
    if task_type in ("deductive", "deduction_full_info", "deduction_hard"):
        gold_states = re.findall(r"State:\s*([^.]+)\.", gold_solution, re.IGNORECASE)
        if not gold_states:
            return 1.0
        gen_states = re.findall(r"State:\s*([^.]+)\.", gen_text, re.IGNORECASE)
        # position-aligned line match; extra lines are neither rewarded nor penalized (denominator = gold lines)
        n_match = sum(
            1 for g, p in zip(gold_states, gen_states) if _normalize(g) == _normalize(p)
        )
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
    m_p = (
        _graded_process(gen_text, task)
        if graded_process
        else float(_check_process(gen_text, task))
    )
    return 0.8 * m_p + 0.2, m_p, m_a


# ── TraceRL PPO-clip + KL ─────────────────────────────────────────────────────


@torch.no_grad()
def _compute_logp_old(
    model,
    full_ids: torch.Tensor,  # (L,) on device - prompt + generated tokens
    step_map: torch.Tensor,  # (gen_len,) CPU
    prompt_len: int,
    block_length: int,
    mask_id: int = MASK_ID,
    shrink: int = 8,
    chunk: int = 4,  # trajectory steps batched per forward
) -> list:
    """
    Pre-compute log P_old(x_t | noisy_ids_t) for every trajectory step.
    MUST be called with model.eval() and inside torch.no_grad(), BEFORE optimizer.step().
    Returns ``(TrajectoryState, logp_old_cpu)`` entries.

    Shrinking only merges steps within a block. Merging states across block
    boundaries would change the incremental-canvas attention context.
    """
    valid_steps = torch.unique(step_map[step_map >= 0])
    collapse = "none"
    if shrink > 0 and valid_steps.numel() > shrink:
        collapse = max(1, math.ceil(valid_steps.numel() / shrink))
    states = trajectory_states(
        full_ids[:prompt_len],
        full_ids[prompt_len:],
        step_map,
        mask_id,
        block_length,
        canvas="incremental",
        collapse=collapse,
    )
    scored = score_trajectory_states(model, states, chunk=chunk)
    return [
        (state.to("cpu"), score.logp.cpu()) for state, score in zip(states, scored)
    ]


def _tracerl_ppo_backward(
    model,
    step_list: list,  # from _compute_logp_old
    advantage: float,
    normalizer: float,  # n_active rollouts - keeps gradient scale stable
    eps: float = 0.2,
    beta: float = 0.01,
    use_kl_k3: bool = True,  # k3 unbiased KL estimator (default in TraceRL)
    chunk: int = 2,  # trajectory steps batched per forward+backward
) -> tuple:
    """
    PPO-clip + KL loss for one rollout, backward() per chunk of steps.
    Batching steps into one forward is mathematically identical to per-step
    backward (losses are summed; each step's loss only depends on its own row).
    Returns (policy_loss, kl_loss, clip_frac, ratio_mean) averaged over steps.
    """
    n_steps = len(step_list)
    if n_steps == 0:
        return 0.0, 0.0, 0.0, 0.0
    device = next(model.parameters()).device
    adv_t = torch.tensor(advantage, dtype=torch.float32, device=device)

    tot_policy = tot_kl = tot_clip = tot_ratio = 0.0

    i = 0
    while i < n_steps:
        state_length = step_list[i][0].input_ids.numel()
        end = i + 1
        while (
            end < n_steps
            and end - i < chunk
            and step_list[end][0].input_ids.numel() == state_length
        ):
            end += 1
        sub = step_list[i:end]
        device_states = [state.to(device) for state, _ in sub]
        scored = score_trajectory_states(
            model,
            device_states,
            chunk=chunk,
            with_grad=True,
        )
        chunk_loss = None
        for (_, logp_old_cpu), score in zip(sub, scored):
            objective = ppo_clip_objective(
                score.logp,
                logp_old_cpu,
                adv_t,
                clip_eps=eps,
                beta=beta,
                kl_estimator="k3" if use_kl_k3 else "k1",
            )
            step_loss = objective.loss / (normalizer * n_steps)
            chunk_loss = step_loss if chunk_loss is None else chunk_loss + step_loss

            with torch.no_grad():
                tot_policy += objective.policy_loss.item()
                tot_kl += objective.kl_loss.item()
                tot_clip += objective.clip_fraction.item()
                tot_ratio += objective.ratio_mean.item()

        chunk_loss.backward()  # frees this chunk's graph
        i = end

    n = max(n_steps, 1)
    return tot_policy / n, tot_kl / n, tot_clip / n, tot_ratio / n
