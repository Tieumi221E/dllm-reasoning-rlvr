import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Callable, Optional


# -- Recipe definitions ------------------------------------------------------
# Each recipe is a filter over (depth, complexity_tier).
# Mapping (cf. 2605.26934 Figure 2(c)):
#   baseline   = pretraining region (D1-4, T1-2); in-distribution control
#   high_depth = depth extension (D6+, T1-2); depth-only transfer test
#   high_tier  = complexity extension (D1-4, T4+); complexity-only transfer test
#   diagonal   = along D~=T; balanced two-axis coverage
#   full_grid  = uniform over the full 60-cell grid (per-cell cap via MAX_TASKS_PER_CELL)
#   ood_focus  = both-axes OOD corner (D5+, T3+); pure out-of-distribution test
#   medium_depth = competence-boundary region (D4-7; base pass@1 ~12-35%),
#                  maximizes within-group variance (addresses sparse-reward failure)
RECIPE_FILTERS: Dict[str, Callable[[int, int], bool]] = {
    'baseline':     lambda d, t: d <= 4 and t <= 2,
    'd_le4':        lambda d, t: d <= 4,            # D1-4, all tiers
    'high_depth':   lambda d, t: d >= 6,
    'high_tier':    lambda d, t: t >= 4,
    'diagonal':     lambda d, t: abs(d - t) <= 2,
    'full_grid':    lambda d, t: True,
    'ood_focus':    lambda d, t: d >= 5 and t >= 3,
    'medium_depth': lambda d, t: 4 <= d <= 7,
    'd2_3':         lambda d, t: 2 <= d <= 3 and t <= 2,   # highest reachable-headroom region (pass@128 ~0.83)
    'd4_5':         lambda d, t: 4 <= d <= 5 and t <= 2,   # competence-boundary region
    # Exact cell definitions aligned with 2605.26934 Table 16:
    'depth_mid':    lambda d, t: 5 <= d <= 7 and t <= 2,   # Depth-Mid: D5-D7×T1-T2 (6 cells) - single depth axis
    'depth_high':   lambda d, t: 8 <= d <= 10 and t <= 2,  # Depth-High: D8-D10×T1-T2 (6 cells) - deep region, direct
    'depth_uniform':lambda d, t: t <= 2,                   # Depth-Uniform: D1-D10×T1-T2 (20 cells) - full depth axis
    'depth_5_10':   lambda d, t: 5 <= d <= 10 and t <= 2,  # D5-D10×T1-T2 (12 cells) - whole OOD depth band
    'offbase_mix':  lambda d, t: not (d <= 4 and t <= 2),  # Offbase-Mix: everything outside the pretrain block (52 cells) - joint coverage
    # Allocation campaign: single-axis sweeps + diagonal + coverage recipes (D1-6 x T1-6)
    'd12_t12':      lambda d, t: d <= 2 and t <= 2,             # shallow-simple
    'd34_t12':      lambda d, t: 3 <= d <= 4 and t <= 2,        # mid-depth, simple
    'd56_t12':      lambda d, t: 5 <= d <= 6 and t <= 2,        # deep, simple (past boundary)
    'd12_t34':      lambda d, t: d <= 2 and 3 <= t <= 4,        # shallow, mid-complex
    'd12_t56':      lambda d, t: d <= 2 and 5 <= t <= 6,        # shallow, high-complex
    'd34_t34':      lambda d, t: 3 <= d <= 4 and 3 <= t <= 4,   # mid diagonal
    'd16_t12':      lambda d, t: d <= 6 and t <= 2,             # depth-uniform (all depth, simple)
    'd14_t16':      lambda d, t: d <= 4 and t <= 6,             # complexity coverage (in-depth, all complexity)
    'd16_t16':      lambda d, t: d <= 6 and t <= 6,             # full coverage (6x6 grid)
}

# Max tasks kept per (depth, tier) cell for full_grid (~2500/cell, aligned with
# rl_full_coverage per_bucket=2500 in rl_variants_recipes.json)
FULL_GRID_CAP_PER_CELL = 2500


# ── Streaming JSON array iterator ──────────────────────────────────────────

def _iter_raw_json_array(path, chunk_size=4 * 1024 * 1024):
    """Stream top-level items from a JSON array file without loading it all."""
    decoder = json.JSONDecoder()
    with open(path, 'r', encoding='utf-8') as f:
        buf = ''
        while '[' not in buf:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"No JSON array found in {path}")
            buf += chunk
        buf = buf[buf.index('[') + 1:]

        while True:
            buf = buf.lstrip()
            if not buf:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buf += chunk
                buf = buf.lstrip()
            if buf[0] == ']':
                break
            if buf[0] == ',':
                buf = buf[1:]
                continue
            while True:
                try:
                    obj, end_idx = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[end_idx:]
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk


# -- Graph -> tasks expansion ------------------------------------------------

def expand_graph_to_tasks(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand one graph-level entry into multiple task-level flat dicts.
    Works for eval.json and rl_total.json (same structure).
    """
    meta = {
        'graph_id':        item['graph_id'],
        'depth':           item['depth'],
        'complexity_tier': item.get('complexity_tier', 1),
        'complexity':      (item['complexity']['composite_score']
                            if isinstance(item.get('complexity'), dict)
                            else float(item.get('complexity', 0.0))),
    }
    story        = item['story']
    masked_story = (item.get('mask_info') or {}).get('masked_story', story)
    out = []

    for task_type in ('deductive', 'inductive', 'analogy'):
        for task in item.get('tasks', {}).get(task_type, []):
            out.append({
                'story':              story,
                'question':           task['question'],
                'solution':           task['solution'],
                'answer':             task['answer'],
                'equivalent_answers': task.get('equivalent_answers', []),
                'task_type':          task_type,
                **meta,
            })

    # abductive tasks use the masked story
    for task in item.get('tasks', {}).get('abductive', []):
        out.append({
            'story':              masked_story,
            'question':           task['question'],
            'solution':           task['solution'],
            'answer':             task['answer'],
            'equivalent_answers': task.get('equivalent_answers', []),
            'task_type':          'abductive',
            **meta,
        })

    return out


# -- Pretrain streaming conversion -------------------------------------------

def _expand_raw_item(item):
    """Expand one raw graph item into a list of flat QA dicts."""
    meta = {
        'graph_id': item['graph_id'],
        'depth': item['depth'],
        'complexity': item['complexity']['composite_score'],
    }
    tasks = item.get('tasks') or {}
    samples = []

    for task_type in ('deductive', 'inductive', 'analogy'):
        for task in tasks.get(task_type, []):
            samples.append({
                'story': item['story'],
                'question': task['question'],
                'answer': task['solution'],
                'task_type': task_type,
                **meta,
            })

    masked_story = (item.get('mask_info') or {}).get('masked_story', item['story'])
    for task in tasks.get('abductive', []):
        samples.append({
            'story': masked_story,
            'question': task['question'],
            'answer': task['solution'],
            'task_type': 'abductive',
            **meta,
        })

    return samples


def convert_json_to_jsonl(json_path, jsonl_path):
    print(f"Converting {json_path} → {jsonl_path} (streaming)...", flush=True)
    tmp_path = jsonl_path + '.tmp'
    count = 0
    with open(tmp_path, 'w', encoding='utf-8') as fout:
        for raw_item in _iter_raw_json_array(json_path):
            for sample in _expand_raw_item(raw_item):
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                count += 1
            if count % 100_000 == 0:
                print(f"  {count:,} QA pairs written...", flush=True)
    os.replace(tmp_path, jsonl_path)
    print(f"Conversion complete: {count:,} QA pairs → {jsonl_path}")
    return count


def prepare_bin(jsonl_path, bin_prefix, tokenizer, max_length=2048):
    bin_path      = bin_prefix + '.bin'
    offsets_path  = bin_prefix + '.offsets.npy'
    lengths_path  = bin_prefix + '.lengths.npy'

    if os.path.exists(bin_path) and os.path.exists(offsets_path):
        offsets = np.load(offsets_path)
        lengths = np.load(lengths_path)
        print(f"Loaded binary dataset: {len(offsets):,} samples")
        return bin_path, offsets, lengths

    print(f"Pre-tokenizing {jsonl_path} → {bin_path} ...", flush=True)
    tmp_bin = bin_path + '.tmp'
    offsets_list = []
    lengths_list = []
    pos = 0

    with open(jsonl_path, 'r', encoding='utf-8') as fin, \
         open(tmp_bin, 'wb') as fbin:
        for i, line in enumerate(fin):
            item = json.loads(line)
            text = (f"Story: {item['story']} "
                    f"Question: {item['question']} "
                    f"Answer: {item['answer']}")
            ids = tokenizer.encode(text)
            if len(ids) > max_length:
                ids = ids[:max_length - 1] + [tokenizer.eos_token_id]
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(fbin)
            offsets_list.append(pos)
            lengths_list.append(len(ids))
            pos += len(ids)
            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:,} samples tokenized...", flush=True)

    offsets = np.array(offsets_list, dtype=np.int64)
    lengths = np.array(lengths_list, dtype=np.int32)
    np.save(offsets_path, offsets)
    np.save(lengths_path, lengths)
    os.replace(tmp_bin, bin_path)
    print(f"Binary ready: {len(offsets):,} samples, {pos:,} tokens → {bin_path}", flush=True)
    return bin_path, offsets, lengths


# -- RL data: stream-expand rl_total.json -> rl_expanded.jsonl ---------------

def prepare_rl_jsonl(
    rl_json_path: str,
    rl_jsonl_path: str,
    recipe_filter: Optional[Callable[[int, int], bool]] = None,
    max_per_cell: Optional[int] = None,
) -> int:
    """
    Stream-expand rl_total.json (98GB) into rl_expanded.jsonl (one task per line).
    recipe_filter(depth, tier) -> bool: keep only matching tasks (None = keep all).
    max_per_cell: cap of tasks per (depth, complexity_tier) cell; None = unlimited.
      For full_grid: prevents abundant low-(D,T) cells from dominating training.
    Atomic rename on success; no truncated file is left on kill -9.
    """
    if os.path.exists(rl_jsonl_path):
        count = sum(1 for _ in open(rl_jsonl_path, encoding='utf-8'))
        print(f"RL JSONL exists: {count:,} tasks -> {rl_jsonl_path}")
        return count

    print(f"Expanding RL data {rl_json_path} -> {rl_jsonl_path} ...", flush=True)
    tmp_path = rl_jsonl_path + '.tmp'
    count = 0
    cell_counts: Dict[tuple, int] = {}   # (depth, tier) → tasks written so far
    with open(tmp_path, 'w', encoding='utf-8') as fout:
        for graph in _iter_raw_json_array(rl_json_path):
            depth = graph.get('depth', 1)
            tier  = graph.get('complexity_tier', 1)
            if recipe_filter is not None and not recipe_filter(depth, tier):
                continue
            if max_per_cell is not None:
                cell_key = (depth, tier)
                if cell_counts.get(cell_key, 0) >= max_per_cell:
                    continue
            tasks_for_graph = list(expand_graph_to_tasks(graph))
            for task in tasks_for_graph:
                fout.write(json.dumps(task, ensure_ascii=False) + '\n')
                count += 1
            if max_per_cell is not None:
                cell_counts[cell_key] = cell_counts.get(cell_key, 0) + len(tasks_for_graph)
            if count % 200_000 == 0 and count > 0:
                print(f"  {count:,} tasks written...", flush=True)
    os.replace(tmp_path, rl_jsonl_path)
    print(f"RL JSONL ready: {count:,} tasks → {rl_jsonl_path}", flush=True)
    return count


# ── Dataset ──────────────────────────────────────────────────────────────────

class KGReasoningDataset(Dataset):
    """Pretrain mode: memmap binary; eval mode: expand all tasks, random access."""

    def __init__(self, file_path, tokenizer, max_length=2048, mode="pretrain"):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.mode       = mode

        if mode == "pretrain":
            if file_path.endswith('.json'):
                base = file_path[:-5]
            else:
                base = file_path.replace('_expanded.jsonl', '').replace('.jsonl', '')
            jsonl_path = base + '_expanded.jsonl'
            bin_prefix = base + '_pretokenized'

            if not os.path.exists(jsonl_path):
                if file_path.endswith('.json') and os.path.exists(file_path):
                    convert_json_to_jsonl(file_path, jsonl_path)
                else:
                    raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

            self._bin_path, self._offsets, self._lengths = prepare_bin(
                jsonl_path, bin_prefix, tokenizer, max_length
            )
            self._mmap = None

        elif mode == "eval":
            # eval.json: graph-level -> expand to task-level list
            with open(file_path, 'r', encoding='utf-8') as f:
                graphs = json.load(f)
            self.tasks: List[Dict] = []
            for graph in graphs:
                self.tasks.extend(expand_graph_to_tasks(graph))
            print(f"Eval dataset: {len(self.tasks):,} tasks from {len(graphs):,} graphs")

        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'pretrain' or 'eval'.")

    def __len__(self):
        if self.mode == "pretrain":
            return len(self._offsets)
        return len(self.tasks)

    def __getitem__(self, idx):
        if self.mode == "pretrain":
            if self._mmap is None:
                self._mmap = np.memmap(self._bin_path, dtype=np.uint16, mode='r')
            offset = int(self._offsets[idx])
            length = int(self._lengths[idx])
            return self._mmap[offset:offset + length].tolist()
        # eval mode: return the task dict
        return self.tasks[idx]


class KGRLStreamingDataset:
    """
    Stream over rl_expanded.jsonl (already recipe-filtered).
    An iterator, not a torch Dataset (no random-access DataLoader).
    The RL loop uses iter(dataset) directly.
    """

    def __init__(self, jsonl_path: str):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"RL JSONL not found: {jsonl_path}\n"
                f"Run prepare_rl_jsonl() first to generate it."
            )
        self.jsonl_path = jsonl_path
        # fast line count
        self._len: Optional[int] = None

    def __len__(self) -> int:
        if self._len is None:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                self._len = sum(1 for _ in f)
        return self._len

    def __iter__(self):
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)


def collate_fn_rl(batch):
    return {
        "prompts":   [x["prompt"] for x in batch],
        "responses": [x["response"] for x in batch],
        "metadata":  [{"graph_id": x["graph_id"], "depth": x["depth"],
                       "complexity": x["complexity"]} for x in batch],
    }
