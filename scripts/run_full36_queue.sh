#!/bin/bash
# Sequential full36 eval queue on one GPU: D1-6 x T1-6 (36 cells), n=128, 40 tasks/cell.
# Matches the original full36_* eval config (gen384/block32/steps32/temp1.0, samples auto-dumped).
# Usage: run_full36_queue.sh <GPU> "name|model_path" "name|model_path" ...
GPU=$1; shift
for spec in "$@"; do
  IFS='|' read -r name model <<< "$spec"
  echo "[$(date +%H:%M:%S)] GPU$GPU start $name ($model)"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n dllm --no-capture-output python -m src.small_evaluate \
    --model_path "$model" --out_path "results/$name.json" \
    --task_type deductive --min_depth 1 --max_depth 6 --min_tier 1 --max_tier 6 \
    --num_samples 128 --max_tasks_per_cell 40 --gen_length 384 --block_length 32 --steps 32 \
    --temperature 1.0 --micro_batch 64 > "logs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$GPU done $name"
done
echo "QUEUE_GPU${GPU}_DONE"
