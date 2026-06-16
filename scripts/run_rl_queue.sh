#!/bin/bash
# Sequential RL-training queue on one GPU for the allocation campaign.
# Usage: run_rl_queue.sh <GPU> <recipe> [<recipe> ...]
GPU=$1; shift
SFT=checkpoints/small_sft_ded_semiar2/step_1800
for r in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU python -m src.small_train_rl \
    --sft_path "$SFT" --out_dir "checkpoints/small_rl_$r" --recipe "$r" \
    --total_steps 1000 --save_every 500 --G 8 --B 4 --lr 5e-6 --beta 0.05 --eps 0.2 \
    --gen_length 384 --block_length 32 --steps_per_block 32 --temperature 1.0 \
    > "logs/rl_$r.log" 2>&1
  echo "$r done"
done
echo "RL_QUEUE_GPU${GPU}_DONE"
