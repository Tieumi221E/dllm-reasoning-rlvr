# Small-dLLM RLVR: depth × complexity allocation

This repository studies where reinforcement learning with verifiable rewards
(RLVR) helps a 105.6M-parameter, from-scratch masked-diffusion language model
on synthetic knowledge-graph reasoning.

The released result is a matched-budget campaign: one supervised base and ten
RL data-allocation recipes, each trained for 1000 optimizer steps and evaluated
on the same D1–6 × T1–6 grid. See [REPORT.md](REPORT.md) for exact settings,
tables, limitations, and figures.

## Main findings

- Depth reduces the base model's `pass@128` ceiling more sharply than
  complexity.
- RLVR gains depend on allocation; no single recipe dominates the grid.
- The strongest in-region result is `d12_t56` (SG +0.091, CG +0.156).
- Off-region transfer is common: eight of ten recipes peak outside their
  training region at the cell level.

## Attribution

| Prior work | Reused idea |
|---|---|
| [Zhu et al., arXiv:2605.26934](https://arxiv.org/abs/2605.26934) | Knowledge-graph task families, D×T allocation grid, verifiable reward, and pass@k analysis |
| [TraceRL / dLLM-RL, arXiv:2509.06949](https://arxiv.org/abs/2509.06949) | Trajectory-decomposed diffusion-LM PPO |
| [LLaDA, arXiv:2502.09992](https://arxiv.org/abs/2502.09992) | Masked-diffusion objective and block-wise generation |

The task generator and generated dataset are not redistributed. This
repository implements the experiment-specific training, reward, evaluation,
and allocation pipeline.

## Shared diffusion core

The code imports `dllm`, the masked-diffusion toolkit maintained in
[Tieumi221E/dllm](https://github.com/Tieumi221E/dllm). It is project source
code, not a PyPI requirement. The released experiment used
`e22684e48a6a4e2637f5112bbaff508b125c7643`; the current code is verified
against `e7b8543f1a68eb2e8476b54bd0121b43aee39b9c` (`dllm` 1.3.2):

```bash
git clone https://github.com/Tieumi221E/dllm.git
git -C dllm checkout e7b8543f1a68eb2e8476b54bd0121b43aee39b9c
python -m pip install -e ./dllm
python -m pip install -r requirements.txt
```

## Repository layout

```text
src/
  model_wrapper.py     model integration
  tokenizer_utils.py   KG tokenizer
  small_sft_semiar.py  block-wise semi-AR SFT
  small_train_rl.py    TraceRL/GRPO training
  small_evaluate.py    paired grid evaluation
  data_utils.py        task expansion and recipe filters
  scoring.py           process/answer scoring and pass@k
  rl_core.py           reward and shared trajectory-PPO adapter
scripts/
  run_rl_queue.sh      final 10-recipe training queue
  run_full36_queue.sh  final full-grid evaluation queue
  campaign_report.py   final figures and aggregate tables
report_assets/         released aggregate tables and figures
```

## Reproduce

Prepare the task JSONL with the generator described by Zhu et al., install the
shared core above, then run:

```bash
python -m src.small_sft_semiar \
  --model_path <pretrained_backbone> \
  --train_data <expanded_tasks.jsonl> \
  --out_dir checkpoints/small_sft_ded_semiar2 \
  --max_depth 4 --task_type deductive

bash scripts/run_rl_queue.sh <GPU> \
  d14_t12 d12_t12 d34_t12 d56_t12 d12_t34 \
  d12_t56 d34_t34 d16_t12 d14_t16 d16_t16

bash scripts/run_full36_queue.sh <GPU> \
  "full36_d14t12_1000|checkpoints/small_rl_d14_t12/step_1000"

python scripts/campaign_report.py
```

The queue scripts expose all fixed training and evaluation settings. Raw data,
checkpoints, logs, and per-sample generations remain intentionally excluded
from Git.

## License

MIT. See [LICENSE](LICENSE).
