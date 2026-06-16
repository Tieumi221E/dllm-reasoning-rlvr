"""
Generation eval for the small diffusion LM (SFT checkpoints), driven by the
debugged LLaDA block generation + reused m_P/m_A text scorers.

Per (depth,tier) cell reports: pass@1, pass@8, gap, m_A, m_P, and the WITHIN-TASK
reward std (the GRPO signal). Dumps EVERY generation to <out>.samples.json and
prints instances (correct / answer-right-trace-wrong / etc.) for eyeballing
format quality, real reasoning errors vs corruption/truncation.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n dllm --no-capture-output python -m src.small_evaluate \
      --model_path checkpoints/small_sft_ded/step_400 \
      --out_path   results/small_sft_s400_d1-4.json \
      --min_depth 1 --max_depth 4 --gen_length 384 --num_samples 8
"""
import argparse, json, math, os, re, sys
import torch

from src.model_wrapper import DiffusionTransformerLM
from src.tokenizer_utils import KGTokenizer
from diffusion_core.inference import DiffusionSampler       # TRUE incremental block gen
from src.scoring import _check_answer, _check_process, _passk  # m_P / m_A / pass@k
from src.data_utils import expand_graph_to_tasks            # noqa: E402

CORRUPT = re.compile(r"brokenact|intactbrok|brokenbroken|actintact|"
                     r"\b(\w+)\s+\1\s+\1\b")


def reward_of(mp, ma):
    return (0.8 * mp + 0.2 * ma) if ma else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--val_data", default="data_generation/output_stratified_pretrain_3200000_0.1562_rl1x_1200000_eval_4800_0.3722_20260527_193136/eval.json")
    ap.add_argument("--task_type", default="deductive")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_tasks_per_cell", type=int, default=4)
    ap.add_argument("--gen_length", type=int, default=384)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=0, help="0 = 16 steps/block")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min_depth", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=0)
    ap.add_argument("--min_tier", type=int, default=0)
    ap.add_argument("--max_tier", type=int, default=0)
    ap.add_argument("--micro_batch", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = KGTokenizer.from_file(os.path.join(args.model_path, "tokenizer.json"))
    model = DiffusionTransformerLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device).eval()
    mask_id, eos_id = tok.mask_token_id, tok.eos_token_id
    # TRUE incremental block generation: blocks are appended one at a time and the
    # model only ever sees [committed prefix + current block] — no fixed full-mask
    # canvas, so gen_length is just a hard cap; length emerges from EOS. This is
    # what the semi-AR SFT was trained for.
    sampler = DiffusionSampler(model.model, tok, mask_id, device, temperature=args.temperature)
    steps_per_block = args.steps or 16
    print(f"model {args.model_path}  MASK={mask_id} EOS={eos_id}  block-gen "
          f"block={args.block_length} steps/block={steps_per_block} cap={args.gen_length} T={args.temperature}")

    # collect tasks per cell
    graphs = json.load(open(args.val_data))
    per_cell = {}
    for g in graphs:
        for t in expand_graph_to_tasks(g):
            if t.get("task_type") != args.task_type or not t.get("solution"):
                continue
            d, tier = t.get("depth"), t.get("complexity_tier")
            if args.min_depth and d < args.min_depth: continue
            if args.max_depth and d > args.max_depth: continue
            if args.min_tier and tier < args.min_tier: continue
            if args.max_tier and tier > args.max_tier: continue
            per_cell.setdefault((d, tier), [])
            if len(per_cell[(d, tier)]) < args.max_tasks_per_cell:
                per_cell[(d, tier)].append(t)
    tasks = [t for v in per_cell.values() for t in v]
    print(f"tasks: {len(tasks)} over {len(per_cell)} cells")

    dump, results = [], []
    for i, t in enumerate(tasks):
        prompt_ids = tok.encode(f"Story: {t['story']} Question: {t['question']} Answer:")[:-1]
        texts = []
        rem = args.num_samples
        while rem > 0:
            mb = min(args.micro_batch, rem)
            gen_lists = sampler.generate_batch_blocked(
                prompt_ids, num_samples=mb, max_new_tokens=args.gen_length,
                block_size=args.block_length, steps_per_block=steps_per_block,
                remask_mode="low_confidence_static", temperature=args.temperature)
            texts += [tok.decode(ids, skip_special_tokens=True) for ids in gen_lists]
            rem -= mb
        gold = t.get("answer", "")
        mps = [bool(_check_process(s, t)) for s in texts]
        mas = [bool(_check_answer(s, gold, t.get("equivalent_answers"))) for s in texts]
        rs  = [reward_of(mp, ma) for mp, ma in zip(mps, mas)]
        nc  = sum(1 for mp, ma in zip(mps, mas) if mp and ma)   # strict correct
        import statistics as st
        d, tier = t.get("depth"), t.get("complexity_tier")
        results.append(dict(depth=d, tier=tier, n_correct=nc, n_total=args.num_samples,
                            m_A=sum(mas)/len(mas), m_P=sum(mps)/len(mps),
                            rstd=st.pstdev(rs),
                            pass_k={str(k): _passk(nc, args.num_samples, k)
                                    for k in (1, 4, 8, 16, 32, 64, 128) if k <= args.num_samples}))
        dump.append(dict(depth=d, tier=tier, gold=gold, question=t["question"][:160],
                         mp=mps, ma=mas, samples=texts))
        print(f"  task{i} D{d}T{tier} strict={nc}/{args.num_samples} "
              f"m_A={results[-1]['m_A']:.2f} m_P={results[-1]['m_P']:.2f} rstd={results[-1]['rstd']:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    json.dump(results, open(args.out_path, "w"), indent=2, ensure_ascii=False)
    json.dump(dump, open(args.out_path + ".samples.json", "w"), indent=2, ensure_ascii=False)

    # ── cell + overall summary ──
    from collections import defaultdict
    cells = defaultdict(lambda: defaultdict(list))
    for r in results:
        c = cells[(r["depth"], r["tier"])]
        pk = r["pass_k"]; c["p1"].append(pk["1"]); c["p8"].append(pk.get("8", pk.get("4", pk["1"])))
        c["ma"].append(r["m_A"]); c["mp"].append(r["m_P"]); c["rstd"].append(r["rstd"])
    print(f"\n{'cell':>8} | {'p@1':>5} {'p@8':>5} {'gap':>6} | {'m_A':>5} {'m_P':>5} {'rstd':>5}")
    g = lambda x: sum(x)/len(x)
    allp1, allp8, allg = [], [], []
    for k in sorted(cells):
        c = cells[k]; p1, p8 = g(c["p1"]), g(c["p8"])
        allp1.append(p1); allp8.append(p8); allg.append(p8-p1)
        print(f"D{k[0]}T{k[1]:>2} | {p1:5.2f} {p8:5.2f} {p8-p1:+6.2f} | {g(c['ma']):5.2f} {g(c['mp']):5.2f} {g(c['rstd']):5.2f}")
    print("-"*52)
    print(f"{'MEAN':>8} | {g(allp1):5.2f} {g(allp8):5.2f} {g(allg):+6.2f} |")

    # ── instances: 1 correct, 1 answer-right/trace-wrong, 1 fully-wrong ──
    print("\n===== INSTANCES =====")
    shown = {"correct": 0, "ans_ok_trace_wrong": 0, "wrong": 0}
    for dd in dump:
        for s, mp, ma in zip(dd["samples"], dd["mp"], dd["ma"]):
            cat = ("correct" if (mp and ma) else
                   "ans_ok_trace_wrong" if (ma and not mp) else "wrong")
            if shown[cat] == 0:
                tagc = "CORRUPT" if CORRUPT.search(s) else ""
                print(f"\n[{cat} {tagc} D{dd['depth']}T{dd['tier']} gold={dd['gold']!r}]")
                print(f"Q: {dd['question'][:100]}")
                print(s[:550])
                shown[cat] = 1
        if all(shown.values()):
            break


if __name__ == "__main__":
    main()
