# Small-dLLM RLVR: a controlled depth × complexity study

Can **RL with verifiable rewards (RLVR)** extend the reasoning competence of a small,
**from-scratch diffusion language model (~107M)** on knowledge-graph reasoning — and where
does it help? We port the AR-baseline RLVR data-allocation study to a discrete diffusion LM
and measure sharpening (Δpass@1) and ceiling gain (Δpass@128) across a depth × complexity grid.

**Headline findings** (details and figures in [`REPORT.md`](REPORT.md)):
- Depth and complexity are **not interchangeable**: depth removes *capability* (pass@128 → 0),
  while complexity removes *reliability* (pass@1 ↓ but pass@128 plateaus at a non-zero floor).
- RLVR produces a **real, replicated, budget-monotone** gain — both sharpening (pass@1) and a
  genuine ceiling gain (pass@128) — where there is headroom *and* a non-trivial base foothold.
- The strongest single effect is **off-diagonal transfer**: training on deep-but-simple data
  lifts shallow-but-complex performance, while direct boundary training fails (a sparse-foothold
  wall, not a ceiling). These match the AR-baseline study, so they are cross-architecture.

---

## Attribution — what is reused vs. what is ours

This project deliberately builds on three pieces of prior work. They are **referenced, not
claimed as ours**:

| Prior work | What we take from it | Where it lives here |
|---|---|---|
| **Zhu et al., [2605.26934](https://arxiv.org/abs/2605.26934)** | Experimental design: the KG task families, the depth × complexity (D×T) grid, the GRPO reward `R = 0.8·m_P + 0.2·m_A`, and the `pass@1` / `pass@128` → SG / CG evaluation. **The synthetic KG dataset is produced by the authors' own generator code.** | the authors' data-generation code — **not redistributed here**; obtain it from the authors. Our pipeline only consumes its output. |
| **TraceRL / dLLM-RL, [2509.06949](https://arxiv.org/abs/2509.06949)** | The RL algorithm for diffusion LMs: `step_map` trajectory-decomposed PPO-clip with no value model. | the authors' reference code is **not included**; our implementation in `src/rl_core.py` (`compute_reward`, `_tracerl_ppo_backward`, `_compute_logp_old`) |
| **LLaDA, [2502.09992](https://arxiv.org/abs/2502.09992)** | The masked-diffusion (MDM) training objective, EOS/padding handling, and block-wise generation. | `diffusion_core/` (our small-model implementation) |

**Our contribution:**
- Applying this full stack to a **small, from-scratch ~107M diffusion LM** (not an 8B model).
- A **block-wise semi-AR SFT** that makes multi-block generation work (`src/small_sft_semiar.py`).
- A **controlled depth × complexity RLVR study** on this model, with paired-`t` significance,
  replication across runs, and budget curves (`src/small_*`, `scripts/`).
- The findings above: the depth/complexity capability-vs-reliability dissociation, the confirmed
  sharpening-plus-ceiling-gain regime, the off-diagonal transfer, and the sparse-foothold wall.

---

## Repository layout

```
diffusion_core/        small discrete-diffusion LM (GQA) + block-wise sampler   [LLaDA-style]
src/
  model_wrapper.py     HF wrapper around the diffusion transformer
  tokenizer_utils.py   KGTokenizer
  small_sft_semiar.py  block-wise semi-AR SFT            (ours)
  small_train_rl.py    TraceRL/GRPO on the small model   (ours; reuses the RL core)
  small_evaluate.py    grid eval, pass@k, m_P/m_A         (ours)
  data_utils.py        task expansion + (D,T) recipe filters
  scoring.py           m_P / m_A scorers + unbiased pass@k
  rl_core.py           gated reward + TraceRL PPO core     [algorithm: 2509.06949]
scripts/
  run_eval_queue.sh    multi-GPU grid-eval queue (3-recipe stage)
  run_full36_queue.sh  multi-GPU grid-eval queue (10-recipe campaign)
  run_rl_queue.sh      multi-GPU RL-training queue
  final_report.py      figures + tables for the 3-recipe stage (REPORT.md §1-7)
  campaign_report.py   figures + tables for the 10-recipe campaign (REPORT.md §8)
report_assets/         report figures + tables (incl. campaign_tables.txt, fig_best_recipe.png)
REPORT.md              the stage report (setup, results, analysis)
```

## Reproduce

```bash
# 1. SFT (block-wise semi-AR) on the prepared region D1-4 × T1-2
# --model_path: a pretrained backbone checkpoint (see "Attribution" above — not redistributed here)
# --train_data: your expanded task JSONL (see src/data_utils.py: prepare_rl_jsonl / expand_graph_to_tasks)
python -m src.small_sft_semiar --model_path <pretrained_backbone> \
    --train_data <expanded_tasks.jsonl> --out_dir checkpoints/small_sft_ded_semiar2 \
    --max_depth 4 --task_type deductive

# 2. RL (TraceRL/GRPO) from the SFT base, per (D,T) recipe
python -m src.small_train_rl --sft_path checkpoints/small_sft_ded_semiar2/step_1800 \
    --recipe baseline --out_dir checkpoints/small_rl_baseline_lr5e6 \
    --lr 5e-6 --G 8 --B 4 --beta 0.05 --gen_length 384 --block_length 32 --steps_per_block 32

# 3. Full-grid eval (D1-6 × T1-6, n=128) for base + each RL checkpoint
bash scripts/run_eval_queue.sh <GPU> "name|checkpoint_path"

# 4. Figures + tables (3-recipe stage)
python scripts/final_report.py

# 3b. 10-recipe campaign: full-grid eval per recipe checkpoint
bash scripts/run_full36_queue.sh <GPU> "name|checkpoint_path" ...

# 4b. Figures + tables (10-recipe campaign)
python scripts/campaign_report.py
```

Reward ≠ eval: the RL reward is the gated `0.8·m_P + 0.2·m_A` (values {0, 0.2, 1.0});
evaluation is strict `m_P ∧ m_A` (pass), aligned with 2605.26934.
