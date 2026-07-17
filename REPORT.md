# Stage report - RLVR on a small diffusion LM over a depth × complexity grid

This report consolidates a controlled study of **RL with verifiable rewards (RLVR)** on a
small, from-scratch **discrete diffusion language model (~105.6M)** for knowledge-graph
reasoning. We measure where, and to what extent, RLVR changes the model's success on a
`depth × complexity` (D × T) grid, using paired before/after comparisons against the
supervised base.

All numbers below come from `results/*.json`, quoted inline in the tables throughout; base
capability is in `report_assets/fig_base_capability.png`, and the full 10-recipe campaign
(§8) is in `report_assets/fig_rl_grid.png` / `fig_best_recipe.png` / `campaign_tables.txt`.

---

## 1. Setup

- **Model.** 105.6M-parameter masked-diffusion transformer (GQA, 12 layers, hidden 768,
  2 KV heads, vocab 3279), trained from scratch - masked-diffusion objective and block-wise
  generation following LLaDA [3]. See the `dllm` package, `src/model_wrapper.py`.
- **Task & data.** Knowledge-graph reasoning with a `depth × complexity_tier` grid; the task
  families, the grid, the reward `R = 0.8·m_P + 0.2·m_A`, and the `pass@1` / `pass@128`
  evaluation follow the AR-baseline study [1]. Supervised training (the "prepared region") uses
  **depth 1-4 × tier 1-2**. The synthetic dataset is produced by the authors' own generator
  code (from [1]), which we do not redistribute; our pipeline only consumes its output.
- **RL.** TraceRL/GRPO for diffusion LMs - `step_map` trajectory-decomposed PPO-clip with no
  value model, following [2]; group-relative advantage with dynamic sampling (zero-variance
  groups dropped). `lr = 5e-6`, `G = 8`, `B = 4`, `beta = 0.05`, `temperature = 1.0`.
  See `src/small_train_rl.py`, `src/rl_core.py`.
- **Evaluation.** `pass@1` (reliability) and `pass@128` (capability ceiling) under strict
  `m_P ∧ m_A`, `n = 128` samples, 40 tasks/cell, `gen_length = 384` (covers every gold trace in
  D1-6×T1-6, max 368 tokens). Significance via **paired `t`-test** (each task is its own
  control: the per-task RL-base difference). See `src/small_evaluate.py`.

**Reward ≠ evaluation.** The RL reward is the gated `0.8·m_P + 0.2·m_A` (values {0, 0.2, 1.0});
evaluation is strict `m_P ∧ m_A`. They share atoms but differ in composition.

---

## 2. Foundations (verified before drawing conclusions)

**The base is fully supervised.** Across SFT checkpoints 1500 → 3000, in-region `pass@1` /
`pass@128` are flat (D1-4: 0.485/0.731 → 0.489/0.750; `results/sftlad_s*.json`). More SFT
extracts nothing, so the base (step_1800) is a converged starting point and the difficulty
structure below is a genuine capability limit, not under-training.

**RL is run only where it can bootstrap, and to a sufficient budget.** For cells with headroom
and non-trivial base success, the sharpening gain plateaus with budget - D2-3 SG `+0.065`
(≈3.9k tasks) → `+0.080` (≈6.5k) → `+0.084` (≈9.1k); `results/paired_rl{600,1000,1400}.json`.

---

## 3. Finding 1 - depth and complexity are different axes: capability vs reliability

The two nominal "difficulty" axes do **not** behave the same way (`results/full36_sft.json`,
`fig_base_capability.png`). Reading the `pass@128` (capability ceiling) marginals:

| axis | D1/T1 | D2/T2 | D3/T3 | D4/T4 | D5/T5 | D6/T6 |
|---|---|---|---|---|---|---|
| pass@128 vs **depth** | 0.48 | 0.59 | 0.41 | 0.23 | 0.12 | **0.05** |
| pass@128 vs **complexity** | 0.58 | 0.53 | 0.28 | 0.20 | 0.18 | **0.12** |

- **Depth removes capability:** the ceiling collapses toward zero with depth (D6 ≈ 0.05).
- **Complexity removes reliability, not capability:** the ceiling declines but **plateaus at a
  non-zero floor** (T6 ≈ 0.12; at shallow depth it stays high - e.g. D2 holds `pass@128 ≈ 0.45`
  all the way to T6 while its `pass@1` falls to 0.07). The answer remains reachable with enough
  samples; the model just cannot find it first-try.

So `pass@1` falls along both axes, but only depth caps what the model can reach at all. (An
apparent additive `pass@1` frontier exists, but it is a reliability artifact: `pass@128` does
not follow it, so the axes are not interchangeable.)

---

## 4. Finding 2 - in-distribution RLVR: sharpening plus a real ceiling gain

Paired results at `step_1200`:

| RL run | region | SG (Δpass@1) | CG (Δpass@128) |
|---|---|---|---|
| baseline (D1-4×T1-2) | trained D1-4×T1-2 | +0.041 (t=3.93) | +0.037 (t=2.47) |
| baseline | D2-3×T1-2 | **+0.073 (t=4.77)** | +0.037 (t=1.91) |
| d2_3 (D2-3×T1-2) | D2-3×T1-2 | **+0.061 (t=4.14)** | +0.031 (t=1.91) |

The effect (i) replicates across two independent runs, (ii) grows monotonically with budget
(§2), and (iii) survives a paired `t`-test at `t > 4` - three checks against small-sample noise.
It is **not pure sharpening**: the `pass@128` ceiling also rises significantly in-region
(CG +0.037, t=2.47), so RLVR adds some genuine capability, not only first-try reliability.

---

## 5. Finding 3 - off-diagonal transfer, and a sparse-foothold boundary wall

- **Direct boundary training fails - not for lack of headroom.** `d4_5` (trained on D4-5×T1-2)
  gives **no gain in its own region** (SG -0.012, t=-0.56) although D4 has large headroom
  (`pass@128 - pass@1 ≈ 0.35`). Base `pass@1` there is ~0.13, so successful rollouts are too rare
  for GRPO to form within-group variance - a **sparse-foothold wall**, not a ceiling.
- **Off-diagonal transfer (the strongest single effect).** The same `d4_5` run sharpens, and
  raises the ceiling of, the **complexity-OOD** region D1-3×T3-6: SG **+0.071 (t=6.91)**,
  CG **+0.087 (t=4.74)**, with per-cell CG up to +0.23 (e.g. D2T5 `pass@128` 0.47→0.68). Training
  on *deep-but-simple* transfers to *shallow-but-complex*, in both reliability and capability.
  This survived an audit (identical eval config; index-aligned same-task pairing; inspected
  generations are genuinely correct; `pass@1` also rises so it is not a diversity artifact; the
  effect is *specific* - `d4_5` is worse in D2-3 - so it is not a globally-stronger model).
  A plausible mechanism is shared long-trace, distractor-robust state tracking, but this is a
  hypothesis the present data cannot confirm.
- **Depth-OOD trades ceiling for reliability.** baseline / d2_3 lift `pass@1` one step past the
  boundary (D5-6×T1-2, SG +0.02, t≈2.7) but lower `pass@128` (CG ≈ -0.025).

---

## 6. Allocation: focused ≈ broad

Concentrating the budget on D2-3 (`d2_3`) gives essentially the same D2-3 sharpening as the
broader D1-4 mix (`baseline`): SG +0.061 vs +0.073, both `t > 4`. With dynamic sampling dropping
saturated cells, narrowing the recipe buys no extra in-region gain - consistent with the
"uniform ≥ curriculum" finding of [1].

---

## 7. Relation to the AR-baseline study [1] - the same phenomena, cross-architecture

The AR baseline (a 107M autoregressive model) uses the identical framework - strict
process-verified `pass@k`, `SG = Δpass@1`, `CG = Δpass@128`, cell-level CG heatmaps on the
(D,T) grid - so the comparison is direct. Its reported phenomena match ours point by point:

| phenomenon | AR baseline [1] | here (diffusion) |
|---|---|---|
| depth is the binding axis for deductive | "Deductive is depth-sensitive" | depth caps capability; complexity caps reliability |
| gains track headroom near the competence boundary | "RL gains depend on headroom near the competence boundary"; "Depth-/Complexity-**High** recipes weaker than their **Mid** counterparts" | sharpening only where headroom + foothold; boundary (`d4_5`) is null |
| off-diagonal / single-axis is suboptimal | "Shallow-Mix beats Cmplx-Uniform on depth and Depth-Uniform on complexity" | `d4_5` (deep) transfers to complexity-OOD, beating the shallow runs there |
| uniform ≥ curriculum / single-axis | Findings 1 & 3 | focused ≈ broad (§6) |

The cross-region transfer is therefore **not a diffusion artifact** but a cross-architecture
property of RLVR data allocation. The one element we make explicit beyond [1] is the
**capability-vs-reliability dissociation** (§3): separating `pass@1` from `pass@128` shows *why*
deductive is depth-sensitive.

---

## 8. Allocation campaign (10 recipes)

A systematic sweep that replaces the single `d4_5` probe: 10 training recipes spanning
single-axis depth (`d12/d34/d56_t12`), single-axis complexity (`d12_t34/d12_t56`), a mid
diagonal (`d34_t34`), broad depth/complexity coverage (`d16_t12`, `d14_t16`), full coverage
(`d16_t16`), and the pretrain-region recipe (`d14_t12`) - all evaluated on the same full
D1-6×T1-6 grid (n=128, 40/cell) at matched RL budget (step_1000), paired against the same SFT
base. See `scripts/campaign_report.py`, `report_assets/campaign_tables.txt`,
`report_assets/fig_rl_grid.png`, `report_assets/fig_best_recipe.png`.

**Off-diagonal transfer replicates across recipes, not just `d4_5`.** 8 of 10 recipes have
their single largest capability gain (peak CG cell) at **D2×T5**, mostly *outside* their own
trained region (`campaign_tables.txt`, "OFF-DIAGONAL TRANSFER"). `d34_t34` (trained mid-diagonal
D3-4×T3-4): CG_in -0.006 (null, t=-0.3) vs CG_out +0.016 (t=2.2) - a transfer index of 1.67,
i.e. essentially all its capability gain lands outside where it trained. `d56_t12` (deep-simple)
shows the same asymmetry (CG_in -0.056, t=-2.2; CG_out +0.027, t=2.9; peak at D1×T5). The one
partial exception is `d12_t56` (trained D1-2×T5-6), whose gain stays mostly in-region
(CG_in +0.156, t=5.4 vs CG_out +0.012).

**Best-recipe-per-cell map (`fig_best_recipe.png`).** No single recipe dominates the grid.
Depth-broad/complexity-broad recipes (`d56_t12`, `d12_t56`) win the most cells in the T4-T6
(complexity-OOD) columns; the pretrain-region recipe (`d14_t12`) only wins within D3-D4. Several
cells (D4-D6 × T2-T3) have no recipe beating base at t≥2.0 - the sparse-foothold wall from §5
persists across the whole campaign, not just for `d4_5`.

## 9. Limitations

- Single small model and one task family (deductive).
- The deep/high-complexity region (base ≈ 0) cannot be RL-trained directly; extrapolation into
  it is measured, not trained.
- The 10-recipe campaign uses a single matched budget (step_1000); per-recipe budget-scaling
  curves (as swept for the original 3 recipes in §2) were not repeated at this scale.

## References

1. *Reasoning Depth and Environment Complexity: A Controlled Study of RLVR Data Allocation across
   Logical Reasoning Tasks* - Zhu et al. (`https://arxiv.org/abs/2605.26934`). Task design, D×T grid, GRPO
   reward, pass@1/pass@128, SG/CG.
2. *Revolutionizing Reinforcement Learning for Diffusion Large Language Models* (TraceRL / dLLM-RL)
   - Wang, Yang et al. (`https://arxiv.org/abs/2509.06949`). Trajectory-decomposed PPO for diffusion LMs.
3. *Large Language Diffusion Models* (LLaDA) - Nie, Zhu et al. (`https://arxiv.org/abs/2502.09992`).
   Masked-diffusion training and block-wise generation.

---

*Figures: `report_assets/fig_base_capability.png` (base capability + depth/complexity marginals),
`report_assets/fig_rl_grid.png` (10-recipe campaign SG/CG heatmaps), `report_assets/fig_best_recipe.png`
(best-recipe-per-cell allocation map). Reproduce via `scripts/campaign_report.py` from
`results/full36_*.json`.*
