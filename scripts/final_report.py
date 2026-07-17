#!/usr/bin/env python
"""Figures + tables for REPORT.md, from the 4 full-grid evals (D1-6 x T1-6,
n=128, 40 tasks/cell): results/full36_{sft,base1200,d23_1200,d45_1200}.json
(index-aligned, so SG/CG are clean paired before/after).

Outputs:
  report_assets/fig_base_capability.png  base pass@1 / pass@128 heatmaps + depth/tier marginals
  report_assets/fig_rl_sg_cg.png         per-RL SG and CG heatmaps (paired)
  report_assets/final_tables.txt         numeric tables for the report
"""
import json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "results/full36_sft.json"
RLS = {"baseline (D1-4xT1-2)": ("results/full36_base1200.json", lambda d, t: d <= 4 and t <= 2),
       "d2_3 (D2-3xT1-2)":     ("results/full36_d23_1200.json", lambda d, t: 2 <= d <= 3 and t <= 2),
       "d4_5 (D4-5xT1-2)":     ("results/full36_d45_1200.json", lambda d, t: 4 <= d <= 5 and t <= 2)}
DEPTHS, TIERS = range(1, 7), range(1, 7)
OUT = "report_assets"


def load(path):
    d = json.load(open(path))
    c = defaultdict(list)
    for x in d:
        c[(x["depth"], x["tier"])].append((x["pass_k"]["1"], x["pass_k"].get("128", 0.0)))
    return c


def grid(cells, idx):
    g = np.full((6, 6), np.nan)
    for (dd, tt), v in cells.items():
        g[dd - 1, tt - 1] = np.mean([x[idx] for x in v])
    return g


def tval(x):
    x = np.asarray(x)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def heat(ax, g, title, cmap="viridis", vmin=0, vmax=1.0, fmt="{:.2f}", diverging=False):
    im = ax.imshow(g, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels([f"T{t}" for t in TIERS])
    ax.set_yticks(range(6)); ax.set_yticklabels([f"D{d}" for d in DEPTHS])
    ax.set_title(title, fontsize=10)
    for i in range(6):
        for j in range(6):
            if not np.isnan(g[i, j]):
                col = "black" if diverging else ("white" if g[i, j] < vmax * 0.6 else "black")
                ax.text(j, i, fmt.format(g[i, j]), ha="center", va="center", color=col, fontsize=7)
    return im


def fig_base(base):
    g1, g128 = grid(base, 0), grid(base, 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    heat(axes[0], g1, "BASE  pass@1  (reliability)")
    heat(axes[1], g128, "BASE  pass@128  (capability ceiling)")
    # marginals: capability vs depth (mean over tiers) and vs tier (mean over depths)
    dep = [np.nanmean(g128[d - 1, :]) for d in DEPTHS]
    tie = [np.nanmean(g128[:, t - 1]) for t in TIERS]
    ax = axes[2]
    ax.plot(list(DEPTHS), dep, "-o", color="#d62728", label="vs depth (mean over tiers)")
    ax.plot(list(TIERS), tie, "-s", color="#1f77b4", label="vs tier (mean over depths)")
    ax.set_xlabel("depth  /  tier"); ax.set_ylabel("pass@128")
    ax.set_title("Capability marginals: depth collapses to ~0,\ncomplexity plateaus at a non-zero floor")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Base (SFT-1800) capability  |  D1-6 x T1-6, n=128", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_base_capability.png", dpi=130); plt.close(fig)


def fig_rl(base, out):
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    for row, (name, (path, trained)) in enumerate(RLS.items()):
        rl = load(path)
        for col, (metric, idx) in enumerate([("SG = Δpass@1", 0), ("CG = Δpass@128", 1)]):
            ax = axes[row, col]
            g = np.full((6, 6), np.nan)
            for (dd, tt) in base:
                if (dd, tt) in rl:
                    b = [x[idx] for x in base[(dd, tt)]]; r = [x[idx] for x in rl[(dd, tt)]]
                    m = min(len(b), len(r)); g[dd - 1, tt - 1] = np.mean(np.array(r[:m]) - np.array(b[:m]))
            heat(ax, g, f"{name}\n{metric}", cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                 fmt="{:+.2f}", diverging=True)
            for dd in DEPTHS:
                for tt in TIERS:
                    if trained(dd, tt):
                        ax.add_patch(plt.Rectangle((tt - 1.5, dd - 1.5), 1, 1, fill=False,
                                                   edgecolor="lime", lw=2.5))
            fig.colorbar(axes[row, col].images[0], ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("RLVR effect per recipe - SG (sharpening) vs CG (ceiling gain)\n"
                 "green box = RL training region; red = gain, blue = loss", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130); plt.close(fig)


def tables(base):
    out = open(f"{OUT}/final_tables.txt", "w")
    def p(*a):
        s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
    g1, g128 = grid(base, 0), grid(base, 1)
    p("=" * 60); p("BASE (SFT-1800)  pass@1 / pass@128   [D1-6 x T1-6]"); p("=" * 60)
    p("  D\\T " + " ".join(f"  T{t}      " for t in TIERS))
    for dd in DEPTHS:
        p(f"  D{dd} " + " ".join(
            f" {g1[dd-1,tt-1]:.2f}/{g128[dd-1,tt-1]:.2f} " for tt in TIERS))
    p("\nCapability (pass@128) marginals:")
    p("  vs depth: " + "  ".join(f"D{d}={np.nanmean(g128[d-1,:]):.2f}" for d in DEPTHS))
    p("  vs tier : " + "  ".join(f"T{t}={np.nanmean(g128[:,t-1]):.2f}" for t in TIERS))
    p("\n" + "=" * 60); p("RL vs BASE (paired, @ step_1200)  SG / CG by region"); p("=" * 60)
    regions = [("trained region", None),
               ("in-dist D2-3xT1-2", lambda d, t: 2 <= d <= 3 and t <= 2),
               ("complexity-OOD D1-3xT3-6", lambda d, t: d <= 3 and t >= 3),
               ("depth-OOD D5-6xT1-2", lambda d, t: d >= 5 and t <= 2)]
    for name, (path, trained) in RLS.items():
        rl = load(path); p(f"\n-- {name} --")
        for lab, pred in regions:
            pr = trained if pred is None else pred
            sg, cg = [], []
            for (dd, tt) in base:
                if pr(dd, tt) and (dd, tt) in rl:
                    b1 = [x[0] for x in base[(dd, tt)]]; r1 = [x[0] for x in rl[(dd, tt)]]
                    b2 = [x[1] for x in base[(dd, tt)]]; r2 = [x[1] for x in rl[(dd, tt)]]
                    m = min(len(b1), len(r1))
                    sg += list(np.array(r1[:m]) - np.array(b1[:m]))
                    cg += list(np.array(r2[:m]) - np.array(b2[:m]))
            p(f"   {lab:>26}: SG {np.mean(sg):+.3f}(t={tval(sg):+.2f})  "
              f"CG {np.mean(cg):+.3f}(t={tval(cg):+.2f})  n={len(sg)}")
    out.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    base = load(BASE)
    fig_base(base)
    fig_rl(base, f"{OUT}/fig_rl_sg_cg.png")
    tables(base)
    print(f"wrote {OUT}/fig_base_capability.png, {OUT}/fig_rl_sg_cg.png, {OUT}/final_tables.txt")
