#!/bin/bash
# Sequential eval queue on one GPU, full grid D1-6 x T1-6 (36 cells), pass@128.
# Usage: run_eval_queue.sh <GPU> "name|model" "name|model" ...
GPU=$1; shift
for spec in "$@"; do
  IFS='|' read -r name model <<< "$spec"
  CUDA_VISIBLE_DEVICES=$GPU python -m src.small_evaluate \
    --model_path "$model" --out_path "results/$name.json" \
    --task_type deductive --min_depth 1 --max_depth 6 --min_tier 1 --max_tier 6 \
    --num_samples 128 --max_tasks_per_cell 40 --gen_length 384 --block_length 32 --steps 32 \
    --temperature 1.0 --micro_batch 64 > "logs/$name.log" 2>&1
  echo "$name done"
done
echo "QUEUE_GPU${GPU}_DONE"
