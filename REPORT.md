# Final report: RLVR allocation on a small diffusion LM

This report records the final, matched-budget experiment only: one supervised
base and ten RLVR allocation recipes, all evaluated on the same deductive
depth × complexity grid. Earlier checkpoint sweeps were exploratory and are
not part of the released evidence.

## Experimental setup

- **Model:** 105.6M-parameter masked-diffusion transformer, with 12 layers,
  hidden size 768, 2 KV heads, and vocabulary size 3279.
- **Base:** block-wise semi-autoregressive SFT checkpoint `step_1800`, trained
  on depth 1–4 × tier 1–2.
- **RL:** TraceRL-style trajectory-decomposed PPO/GRPO without a value model;
  `lr=5e-6`, `G=8`, `B=4`, `beta=0.05`, `eps=0.2`,
  `temperature=1.0`, and 1000 optimizer steps per recipe.
- **Generation:** length 384, block length 32, and 32 denoising steps per
  block.
- **Evaluation:** depth 1–6 × tier 1–6, 40 tasks per cell, 128 samples per
  task, with every RL checkpoint paired by task index against the same SFT
  base.
- **Metrics:** strict correctness is `m_P ∧ m_A`; `SG = Δpass@1` measures
  reliability and `CG = Δpass@128` measures capability-ceiling change.
  Reported `t` values are paired task-level statistics.

The RL reward is not the evaluation metric. Training uses the gated reward
`0.8·m_P + 0.2·m_A`, with possible values `{0, 0.2, 1.0}`; evaluation requires
both process and answer correctness.

## Final conditions

The campaign compares the base against these ten `step_1000` checkpoints:

| Recipe | RL training region |
|---|---|
| `d14_t12` | D1–4 × T1–2 |
| `d12_t12` | D1–2 × T1–2 |
| `d34_t12` | D3–4 × T1–2 |
| `d56_t12` | D5–6 × T1–2 |
| `d12_t34` | D1–2 × T3–4 |
| `d12_t56` | D1–2 × T5–6 |
| `d34_t34` | D3–4 × T3–4 |
| `d16_t12` | D1–6 × T1–2 |
| `d14_t16` | D1–4 × T1–6 |
| `d16_t16` | D1–6 × T1–6 |

## Base capability

The SFT base separates depth from complexity. Its `pass@128` marginals are:

| Axis | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Depth | 0.48 | 0.59 | 0.41 | 0.23 | 0.12 | 0.05 |
| Tier | 0.58 | 0.53 | 0.28 | 0.20 | 0.18 | 0.12 |

Depth therefore removes reachable capability more sharply, while high
complexity retains a low but non-zero sampling ceiling. The complete 36-cell
base table is in `report_assets/campaign_tables.txt`.

## Matched-budget results

| Recipe | Trained-region SG | Trained-region CG | Out-region SG | Out-region CG |
|---|---:|---:|---:|---:|
| `d14_t12` | +0.039 (t=3.8) | +0.047 (t=2.8) | +0.010 (t=3.9) | +0.008 (t=1.1) |
| `d12_t12` | -0.003 (t=-0.1) | +0.025 (t=1.2) | +0.010 (t=3.3) | +0.014 (t=2.0) |
| `d34_t12` | +0.036 (t=2.3) | +0.050 (t=2.2) | +0.012 (t=3.3) | +0.012 (t=1.7) |
| `d56_t12` | +0.004 (t=0.2) | -0.056 (t=-2.2) | +0.018 (t=2.8) | +0.027 (t=2.9) |
| `d12_t34` | +0.073 (t=4.5) | +0.075 (t=2.6) | +0.008 (t=1.3) | +0.006 (t=0.8) |
| `d12_t56` | +0.091 (t=5.6) | +0.156 (t=5.4) | +0.016 (t=2.7) | +0.012 (t=1.4) |
| `d34_t34` | +0.021 (t=1.9) | -0.006 (t=-0.3) | +0.024 (t=5.6) | +0.016 (t=2.2) |
| `d16_t12` | +0.004 (t=0.4) | +0.004 (t=0.3) | +0.025 (t=6.5) | +0.030 (t=3.4) |
| `d14_t16` | +0.028 (t=4.1) | +0.041 (t=4.1) | +0.004 (t=0.7) | -0.006 (t=-0.7) |
| `d16_t16` | +0.023 (t=4.7) | +0.025 (t=3.3) | n/a | n/a |

The clearest in-region gain is `d12_t56`: SG +0.091 and CG +0.156. Broad
full-grid training also gives smaller but positive aggregate gains
(`d16_t16`: SG +0.023, CG +0.025).

## Allocation findings

1. **No recipe dominates the grid.** The best allocation depends on the test
   cell; `report_assets/fig_best_recipe.png` applies a paired `t≥2` gate.
2. **Transfer is often off-region.** Eight of ten recipes obtain their
   largest cell-level CG outside their training region. For example,
   `d34_t34` has trained-region CG -0.006 but out-region CG +0.016, while
   `d56_t12` has -0.056 in-region and +0.027 out-region.
3. **D2×T5 is a recurring transfer target.** Seven recipes peak there,
   including `d14_t12`, `d34_t12`, `d12_t34`, `d34_t34`, `d16_t12`,
   `d14_t16`, and `d16_t16`.
4. **Sparse-foothold regions remain difficult.** Several D4–D6 × T2–T3 cells
   have no recipe with a positive, paired `t≥2` improvement.

These statements are descriptive for this model, task family, and fixed
budget. They do not establish universal scaling behavior.

## Reproducibility boundary

The public repository contains:

- the final analysis script, `scripts/campaign_report.py`;
- aggregate tables and figures under `report_assets/`;
- training, generation, evaluation, reward, and scoring code.

Raw tasks, model checkpoints, logs, and per-sample generations are excluded.
To reproduce the tables, place the SFT aggregate at
`results/full36_sft.json` and the ten matched RL aggregates at
`results/full36_<recipe>_1000.json`, then run:

```bash
python scripts/campaign_report.py
```

## References

1. Zhu et al., *Reasoning Depth and Environment Complexity: A Controlled
   Study of RLVR Data Allocation across Logical Reasoning Tasks*,
   [arXiv:2605.26934](https://arxiv.org/abs/2605.26934).
2. Wang, Yang et al., *Revolutionizing Reinforcement Learning for Diffusion
   Large Language Models*, [arXiv:2509.06949](https://arxiv.org/abs/2509.06949).
3. Nie, Zhu et al., *Large Language Diffusion Models*,
   [arXiv:2502.09992](https://arxiv.org/abs/2502.09992).

## License

MIT. See [LICENSE](LICENSE).
