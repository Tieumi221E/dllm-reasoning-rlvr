"""
TraceRL/GRPO for the small diffusion LM.

The optimisation (GRPO group-z-score advantage, no value model; TraceRL step_map
PPO-clip + k3 KL; dynamic sampling that discards zero-variance groups) is imported
from src.rl_core:
    compute_reward, _compute_logp_old, _tracerl_ppo_backward
Only the model/tokenizer/prompt/rollout/data are specific to the small model:
    - DiffusionTransformerLM (GQA, full fine-tune) + KGTokenizer, MASK_ID=3
    - raw "Story: ... Answer:" prompt (no chat template)
    - rollout = generate_blocked_with_history  (TRUE incremental block gen + step_map)
    - tasks filtered to ONE cell (depth, tier) from the cached deductive task list

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n dllm --no-capture-output python -m src.small_train_rl \
      --sft_path checkpoints/small_sft_ded_semiar2/step_1800 \
      --out_dir  checkpoints/small_rl_depthmid --recipe depth_mid \
      --lr 1e-6 --total_steps 6000 --B 4 --G 8
"""
import argparse, os, pickle, random, time
import torch

from src.model_wrapper import DiffusionTransformerLM
from src.tokenizer_utils import KGTokenizer
from diffusion_core.inference import DiffusionSampler          # same proven generator as eval
from src.rl_core import compute_reward, _compute_logp_old, _tracerl_ppo_backward
from src.data_utils import RECIPE_FILTERS               # reuse the paper-aligned recipe regions

CACHE = "cache/tasks_full_grid_deductive_50000.pkl"


def load_recipe_tasks(recipe, cache=CACHE):
    """Filter the cached deductive task pool by the recipe's (D,T) region —
    the paper-aligned RECIPE_FILTERS (depth_mid = D5-7×T1-2, etc.)."""
    from collections import Counter
    flt = RECIPE_FILTERS[recipe]
    tasks = pickle.load(open(cache, "rb"))
    sel = [t for t in tasks if t.get("solution")
           and flt(int(t.get("depth", 0)), int(t.get("complexity_tier", 0)))]
    cells = Counter((t["depth"], t["complexity_tier"]) for t in sel)
    print(f"recipe={recipe}: {len(sel)} tasks over {len(cells)} cells {sorted(cells)}")
    return sel


def train(args):
    device = torch.device("cuda")
    tok = KGTokenizer.from_file(os.path.join(args.sft_path, "tokenizer.json"))
    mask_id, eos_id = tok.mask_token_id, tok.eos_token_id
    model = DiffusionTransformerLM.from_pretrained(args.sft_path, torch_dtype=torch.bfloat16).to(device)
    sampler = DiffusionSampler(model.model, tok, mask_id, device, temperature=args.temperature)
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.1f}M (full FT)  MASK={mask_id} EOS={eos_id}")

    tasks = load_recipe_tasks(args.recipe)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"TraceRL/GRPO: steps={args.total_steps} G={args.G} B={args.B} lr={args.lr} "
          f"beta={args.beta} eps={args.eps} | rollout: blocked gen={args.gen_length} "
          f"block={args.block_length} steps/blk={args.steps_per_block} T={args.temperature}")

    win_draws = win_accepted = 0
    for opt_step in range(1, args.total_steps + 1):
        t0 = time.time()
        model.eval()
        rollout_data, draws = [], 0
        while len(rollout_data) < args.B and draws < args.max_draws:
            draws += 1
            task = random.choice(tasks)
            pid = tok.encode(f"Story: {task['story']} Question: {task['question']} Answer:")[:-1]
            prompt_ids = torch.tensor([pid], dtype=torch.long, device=device)
            prompt_len = prompt_ids.shape[1]
            # ── rollout: reuse the proven DiffusionSampler (diverse, true block gen) ──
            with torch.no_grad():
                gen = sampler.generate_batch_blocked(
                    pid, num_samples=args.G, max_new_tokens=args.gen_length,
                    block_size=args.block_length, steps_per_block=args.steps_per_block,
                    temperature=args.temperature)
            texts = [tok.decode(r, skip_special_tokens=True) for r in gen]
            # block-based step_map: a token's "denoising step" = its block index
            # (matches the block-wise generation + semi-AR SFT). post-EOS pad = -1.
            g_rows, m_rows = [], []
            for r in gen:
                r = (r + [eos_id])[:args.gen_length]
                nreal = len(r)
                r = r + [eos_id] * (args.gen_length - nreal)
                sm = [i // args.block_length for i in range(nreal)] + [-1] * (args.gen_length - nreal)
                g_rows.append(r); m_rows.append(sm)
            g_ids = torch.tensor(g_rows, dtype=torch.long)        # (G, gen_length) CPU
            g_map = torch.tensor(m_rows, dtype=torch.long)        # (G, gen_length) CPU

            rt = [compute_reward(t, task) for t in texts]
            rewards = torch.tensor([r[0] for r in rt], dtype=torch.float32)
            rewards_mp = torch.tensor([r[1] for r in rt], dtype=torch.float32)
            rewards_ma = torch.tensor([r[2] for r in rt], dtype=torch.float32)
            if rewards.std() < 1e-6:
                continue                                              # zero-variance → resample

            with torch.no_grad():
                step_lists = [_compute_logp_old(
                    model, torch.cat([prompt_ids[0], g_ids[g].to(device)]), g_map[g], prompt_len,
                    mask_id, args.shrink) for g in range(args.G)]
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rollout_data.append(dict(prompt_ids=prompt_ids[0].cpu(), gen_ids=g_ids.cpu(),
                                     step_lists=step_lists, advantages=advantages,
                                     rewards=rewards, rewards_mp=rewards_mp, rewards_ma=rewards_ma))
        torch.cuda.empty_cache()
        win_draws += draws; win_accepted += len(rollout_data)
        if not rollout_data:
            print(f"step={opt_step}/{args.total_steps} no-variance after {draws} draws — skip"); continue

        # ── PPO update (reused TraceRL backward) ──
        model.train(); optimizer.zero_grad()
        n_active = sum(1 for rd in rollout_data for g in range(args.G)
                       if abs(rd["advantages"][g].item()) > 1e-6)
        norm = max(n_active, 1)
        tp = tk = tc = tr = 0.0; n_ppo = 0
        for rd in rollout_data:
            pid = rd["prompt_ids"].to(device)
            for g in range(args.G):
                adv = rd["advantages"][g].item()
                if abs(adv) < 1e-6:
                    continue
                full_ids = torch.cat([pid, rd["gen_ids"][g].to(device)])
                p, k, c, r = _tracerl_ppo_backward(
                    model, full_ids, rd["step_lists"][g], adv, norm,
                    eps=args.eps, beta=args.beta, mask_id=mask_id, chunk=args.ppo_chunk)
                tp += p; tk += k; tc += c; tr += r; n_ppo += 1
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0).item()
        optimizer.step()

        if opt_step % args.log_every == 0:
            R = torch.cat([rd["rewards"] for rd in rollout_data])
            MP = torch.cat([rd["rewards_mp"] for rd in rollout_data])
            MA = torch.cat([rd["rewards_ma"] for rd in rollout_data])
            ns = max(n_ppo, 1)
            print(f"step={opt_step}/{args.total_steps} policy={tp/ns:.4f} kl={tk/ns:.4f} "
                  f"clip={tc/ns:.3f} ratio={tr/ns:.3f} reward={R.mean():.3f}±{R.std():.3f} "
                  f"mp={MP.mean():.3f} ma={MA.mean():.3f} nonzero={(R>0).float().mean():.2f} "
                  f"gnorm={gnorm:.3f} accept={win_accepted}/{win_draws} t={time.time()-t0:.0f}s")
            win_draws = win_accepted = 0
        if opt_step % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"step_{opt_step}")
            model.save_pretrained(ck); tok.save_pretrained(ck)
            print(f"saved → {ck}")

    model.save_pretrained(args.out_dir); tok.save_pretrained(args.out_dir)
    print(f"RL done → {args.out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--recipe", required=True, choices=list(RECIPE_FILTERS.keys()))
    p.add_argument("--total_steps", type=int, default=300)
    p.add_argument("--G", type=int, default=16)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--gen_batch", type=int, default=8)
    p.add_argument("--max_draws", type=int, default=16)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps_per_block", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--shrink", type=int, default=8)
    p.add_argument("--ppo_chunk", type=int, default=2)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=25)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
