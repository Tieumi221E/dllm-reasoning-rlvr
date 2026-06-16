"""
Block-wise semi-autoregressive SFT for the small diffusion LM (corrected design).

The ONLY thing trained, per example: given a CLEAN prefix [prompt + answer blocks
0..k-1], predict the (k+1)-th block — and NOTHING after it is in the sequence, so
there is no future-length leakage (matches TRUE incremental block generation).

  - response (= answer tokens + its EOS) is padded with EOS to a multiple of
    block_length, then cut into blocks of block_length.
  - sample k ∈ [0, num_blocks-1]. sequence = prompt + blocks[0:k] (clean) + block[k].
  - WITHIN block[k] apply standard MDM random masking Bernoulli(t), t~U(0,1):
    high t = cold-start a fresh block, low t = refine a partly-filled block —
    exactly the two things block-diffusion does per block.
  - EOS is the continue/terminate signal: a mid-response block is pure content
    (→ learn to continue, no EOS); the final block holds content-tail + EOS pad
    (→ learn to terminate). e.g. 57-token answer, block 32: k=1 block = 25 content
    + 7 EOS.
  - prompt and the clean prefix blocks are never masked, never in the loss.
"""
import argparse, json, os, random, sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.model_wrapper import DiffusionTransformerLM
from src.tokenizer_utils import KGTokenizer

sys.stdout.reconfigure(line_buffering=True)


def build_example(tok, story, question, answer, block_length, max_seq_len):
    prompt_ids = tok.encode(f"Story: {story} Question: {question} Answer:")[:-1]
    resp = tok.encode(answer)[1:]                                   # content ... [EOS]
    resp += [tok.eos_token_id] * ((-len(resp)) % block_length)      # pad to block edge
    if not resp or len(prompt_ids) + len(resp) > max_seq_len:
        return None
    return prompt_ids, resp, len(resp) // block_length


def make_batch(samples, tok, block_length, device, eps=1e-3):
    eos_id, mask_id = tok.eos_token_id, tok.mask_token_id
    seqs, golds, lmasks, pmasks = [], [], [], []
    for prompt_ids, resp, num_blocks in samples:
        k = random.randrange(num_blocks)
        prefix = prompt_ids + resp[: k * block_length]              # clean
        block  = resp[k * block_length : (k + 1) * block_length]    # target block
        t = random.random() * (1 - eps) + eps
        mask_flags = [random.random() < t for _ in block]
        if not any(mask_flags):
            mask_flags[random.randrange(len(block))] = True
        masked_block = [mask_id if m else tid for tid, m in zip(block, mask_flags)]
        seq  = prefix + masked_block
        gold = prefix + block
        lm   = [False] * len(prefix) + mask_flags                   # loss = masked block pos only
        seqs.append(seq); golds.append(gold); lmasks.append(lm); pmasks.append(t)
    max_len = max(len(s) for s in seqs)
    body, target, attn, loss, pmask = [], [], [], [], []
    for seq, gold, lm, t in zip(seqs, golds, lmasks, pmasks):
        pad = max_len - len(seq)
        body.append(seq + [eos_id] * pad)
        target.append(gold + [eos_id] * pad)
        attn.append([1] * len(seq) + [0] * pad)
        loss.append(lm + [False] * pad)
        pmask.append([t] * max_len)
    to = lambda x, d=torch.long: torch.tensor(x, dtype=d, device=device)
    return to(body), to(target), to(attn), to(loss, torch.bool), to(pmask, torch.float)


def loss_fn(model, body, target, attn, lossmask, pmask):
    logits = model(body, attention_mask=attn).logits
    sel = lossmask
    if not sel.any():
        return None
    ce = F.cross_entropy(logits[sel].float(), target[sel], reduction="none")
    ce = ce / pmask[sel]                                            # MDM importance weight
    seq_idx = sel.nonzero(as_tuple=False)[:, 0]
    B = body.shape[0]
    seq_sum = torch.zeros(B, device=body.device).scatter_add_(0, seq_idx, ce)
    seq_cnt = torch.zeros(B, device=body.device).scatter_add_(
        0, seq_idx, torch.ones_like(ce)).clamp(min=1)
    return (seq_sum / seq_cnt).mean()


def stream_train(jsonl_path, max_depth, task_type, buf_size=8000):
    while True:
        buf = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("depth", 99) > max_depth: continue
                if task_type and d.get("task_type") != task_type: continue
                buf.append(d)
                if len(buf) >= buf_size:
                    random.shuffle(buf); yield from buf; buf = []
        if buf:
            random.shuffle(buf); yield from buf


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = KGTokenizer.from_file(os.path.join(args.model_path, "tokenizer.json"))
    print(f"tokenizer vocab={tok.vocab_size} EOS={tok.eos_token_id} MASK={tok.mask_token_id}")
    model = DiffusionTransformerLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device).train()
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.1f}M  block={args.block_length} (block-wise semi-AR)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * args.max_steps), args.max_steps)
    os.makedirs(args.out_dir, exist_ok=True)
    stream = stream_train(args.train_data, args.max_depth, args.task_type)
    running = 0.0
    optimizer.zero_grad()
    for step in range(1, args.max_steps + 1):
        samples = []
        while len(samples) < args.batch_size:
            t = next(stream)
            ex = build_example(tok, t["story"], t["question"], t["answer"], args.block_length, args.max_seq_len)
            if ex: samples.append(ex)
        body, target, attn, lm, pmask = make_batch(samples, tok, args.block_length, device)
        loss = loss_fn(model, body, target, attn, lm, pmask)
        if loss is None: continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        running += loss.item()
        if step % 20 == 0:
            print(f"step={step}/{args.max_steps} loss={running/20:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
            running = 0.0
        if step % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"step_{step}")
            model.save_pretrained(ck, safe_serialization=False); tok.save_pretrained(ck)
            print(f"saved → {ck}")
    model.save_pretrained(args.out_dir, safe_serialization=False); tok.save_pretrained(args.out_dir)
    print(f"done → {args.out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Pretrained backbone checkpoint dir (see README Attribution)")
    p.add_argument("--train_data", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_type", default="deductive")
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--save_every", type=int, default=300)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--max_seq_len", type=int, default=1280)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
