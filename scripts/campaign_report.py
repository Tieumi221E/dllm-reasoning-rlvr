#!/usr/bin/env python
"""Summary figures + tables for the deductive allocation campaign.

11-condition final set, all full-grid evals (D1-6 x T1-6, n=128, 40 tasks/cell),
index-aligned so SG/CG are clean paired before/after vs the SFT base:

  base : results/full36_sft.json            (small_sft_ded_semiar2/step_1800, no RL)
  10 RL: results/full36_<recipe>_1000.json  (each recipe's step_1000)

Metrics (per cell, paired over the 40 shared tasks):
  SG = Δpass@1   (sharpening / reliability gain)
  CG = Δpass@128 (capability-ceiling gain)

Outputs (report_assets/):
  fig_base_capability.png  base pass@1 / pass@128 + depth/tier marginals
  fig_rl_grid.png          SG & CG heatmap for every recipe (green box = trained region)
  fig_best_recipe.png      best-recipe-per-cell maps (SG and CG), noise-gated by paired t
  campaign_tables.txt      numeric region tables with paired t-tests
"""
import json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "results/full36_sft.json"

# recipe -> (result file, trained-region predicate)   [mirrors data_utils.RECIPE_FILTERS]
RECIPES = {
    "d14_t12": ("results/full36_d14t12_1000.json", lambda d, t: d <= 4 and t <= 2),          # pretrain/SFT region
    "d12_t12": ("results/full36_d12t12_1000.json", lambda d, t: d <= 2 and t <= 2),          # shallow-simple
    "d34_t12": ("results/full36_d34t12_1000.json", lambda d, t: 3 <= d <= 4 and t <= 2),     # mid-depth simple
    "d56_t12": ("results/full36_d56t12_1000.json", lambda d, t: 5 <= d <= 6 and t <= 2),     # deep simple
    "d12_t34": ("results/full36_d12t34_1000.json", lambda d, t: d <= 2 and 3 <= t <= 4),     # shallow mid-complex
    "d12_t56": ("results/full36_d12t56_1000.json", lambda d, t: d <= 2 and 5 <= t <= 6),     # shallow high-complex
    "d34_t34": ("results/full36_d34t34_1000.json", lambda d, t: 3 <= d <= 4 and 3 <= t <= 4),# mid diagonal
    "d16_t12": ("results/full36_d16t12_1000.json", lambda d, t: d <= 6 and t <= 2),          # depth-axis broad
    "d14_t16": ("results/full36_d14t16_1000.json", lambda d, t: d <= 4 and t <= 6),          # complexity-axis broad
    "d16_t16": ("results/full36_d16t16_1000.json", lambda d, t: d <= 6 and t <= 6),          # full coverage
}
DEPTHS, TIERS = range(1, 7), range(1, 7)
T_GATE = 2.0          # paired-t threshold for a cell's best gain to count as a real winner
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


def paired_diff(base, rl, dd, tt, idx):
    """Index-aligned per-task diff (rl - base) for one cell; returns np.array."""
    b = [x[idx] for x in base[(dd, tt)]]
    r = [x[idx] for x in rl[(dd, tt)]]
    m = min(len(b), len(r))
    return np.array(r[:m]) - np.array(b[:m])


def tval(x):
    x = np.asarray(x)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def heat(ax, g, title, cmap="viridis", vmin=0, vmax=1.0, fmt="{:.2f}", diverging=False):
    ax.imshow(g, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels([f"T{t}" for t in TIERS])
    ax.set_yticks(range(6)); ax.set_yticklabels([f"D{d}" for d in DEPTHS])
    ax.set_title(title, fontsize=9)
    for i in range(6):
        for j in range(6):
            if not np.isnan(g[i, j]):
                col = "black" if diverging else ("white" if g[i, j] < vmax * 0.6 else "black")
                ax.text(j, i, fmt.format(g[i, j]), ha="center", va="center", color=col, fontsize=7)


def fig_base(base):
    g1, g128 = grid(base, 0), grid(base, 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    heat(axes[0], g1, "BASE  pass@1  (reliability)")
    heat(axes[1], g128, "BASE  pass@128  (capability ceiling)")
    dep = [np.nanmean(g128[d - 1, :]) for d in DEPTHS]
    tie = [np.nanmean(g128[:, t - 1]) for t in TIERS]
    ax = axes[2]
    ax.plot(list(DEPTHS), dep, "-o", color="#d62728", label="vs depth (mean over tiers)")
    ax.plot(list(TIERS), tie, "-s", color="#1f77b4", label="vs tier (mean over depths)")
    ax.set_xlabel("depth  /  tier"); ax.set_ylabel("pass@128")
    ax.set_title("Capability marginals")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Base (SFT-1800) capability  |  D1-6 x T1-6, n=128", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_base_capability.png", dpi=130); plt.close(fig)


def fig_rl_grid(base, rls):
    """One row per recipe: SG heatmap | CG heatmap, trained region boxed."""
    n = len(RECIPES)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n))
    for row, (name, (rl, trained)) in enumerate(rls.items()):
        for col, (metric, idx) in enumerate([("SG = Δpass@1", 0), ("CG = Δpass@128", 1)]):
            ax = axes[row, col]
            g = np.full((6, 6), np.nan)
            for (dd, tt) in base:
                if (dd, tt) in rl:
                    g[dd - 1, tt - 1] = paired_diff(base, rl, dd, tt, idx).mean()
            heat(ax, g, f"{name}  {metric}", cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                 fmt="{:+.2f}", diverging=True)
            for dd in DEPTHS:
                for tt in TIERS:
                    if trained(dd, tt):
                        ax.add_patch(plt.Rectangle((tt - 1.5, dd - 1.5), 1, 1, fill=False,
                                                   edgecolor="lime", lw=2.0))
    fig.suptitle("RLVR effect per recipe — SG (sharpening) vs CG (ceiling gain)\n"
                 "green box = RL training region; red = gain, blue = loss", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{OUT}/fig_rl_grid.png", dpi=120); plt.close(fig)


def fig_best_recipe(base, rls):
    """For each cell, the recipe maximising the (paired, t-gated) gain. Two panels: SG, CG."""
    names = list(rls.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (metric, idx) in zip(axes, [("SG = Δpass@1", 0), ("CG = Δpass@128", 1)]):
        win_val = np.full((6, 6), np.nan)
        win_lab = [["" for _ in TIERS] for _ in DEPTHS]
        for dd in DEPTHS:
            for tt in TIERS:
                best_m, best_t, best_name = -1e9, 0.0, None
                for name in names:
                    rl, _ = rls[name]
                    if (dd, tt) not in rl:
                        continue
                    diff = paired_diff(base, rl, dd, tt, idx)
                    m, t = diff.mean(), tval(diff)
                    if m > best_m:
                        best_m, best_t, best_name = m, t, name
                # noise gate: require positive gain significant at t>=T_GATE
                if best_name is not None and best_m > 0 and best_t >= T_GATE:
                    win_val[dd - 1, tt - 1] = best_m
                    win_lab[dd - 1][tt - 1] = f"{best_name}\n{best_m:+.2f}"
                else:
                    win_val[dd - 1, tt - 1] = 0.0
                    win_lab[dd - 1][tt - 1] = "—"
        im = ax.imshow(win_val, origin="lower", cmap="YlOrRd", vmin=0, vmax=0.25, aspect="auto")
        ax.set_xticks(range(6)); ax.set_xticklabels([f"T{t}" for t in TIERS])
        ax.set_yticks(range(6)); ax.set_yticklabels([f"D{d}" for d in DEPTHS])
        ax.set_title(f"Best recipe per cell — {metric}\n(blank '—' = no recipe beats base at t≥{T_GATE})",
                     fontsize=10)
        for i in range(6):
            for j in range(6):
                lab = win_lab[i][j]
                col = "black" if win_val[i, j] < 0.15 else "white"
                ax.text(j, i, lab, ha="center", va="center", color=col, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="best paired gain")
    fig.suptitle("Allocation map: which training region wins each test cell "
                 "(paired, noise-gated)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/fig_best_recipe.png", dpi=130); plt.close(fig)


def _summ(sg, cg):
    """Return formatted 'SG(t) CG(t) SG-CG(t) -> label' for a pooled set of paired tasks."""
    sg, cg = np.asarray(sg), np.asarray(cg)
    contrast = sg - cg                              # >0 sharpening-dominant, <0 capability-dominant
    msg, mcg, mct = sg.mean(), cg.mean(), contrast.mean()
    tsg, tcg, tct = tval(sg), tval(cg), tval(contrast)
    # classification (gated at |t|>=2): need a real SG first, then look at the SG-CG contrast / CG
    if tsg < 2 and tcg < 2:
        lab = "null"
    elif tcg >= 2 and (mcg >= msg or tct <= -2):
        lab = "CAPABILITY (ceiling↑)"
    elif msg > 0 and (tcg < 2 or mct > 0):
        lab = "sharpening (reliab↑, ceiling~)"
    else:
        lab = "mixed"
    return (f"SG {msg:+.3f}(t{tsg:+.1f})  CG {mcg:+.3f}(t{tcg:+.1f})  "
            f"SG-CG {mct:+.3f}(t{tct:+.1f})  -> {lab}"), len(sg)


def tables(base, rls):
    out = open(f"{OUT}/campaign_tables.txt", "w")
    def p(*a):
        s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
    g1, g128 = grid(base, 0), grid(base, 1)
    p("=" * 78); p("BASE (SFT-1800)  pass@1 / pass@128   [D1-6 x T1-6]"); p("=" * 78)
    p("  D\\T " + " ".join(f"  T{t}      " for t in TIERS))
    for dd in DEPTHS:
        p(f"  D{dd} " + " ".join(f" {g1[dd-1,tt-1]:.2f}/{g128[dd-1,tt-1]:.2f} " for tt in TIERS))
    p("\nCapability (pass@128) marginals:")
    p("  vs depth: " + "  ".join(f"D{d}={np.nanmean(g128[d-1,:]):.2f}" for d in DEPTHS))
    p("  vs tier : " + "  ".join(f"T{t}={np.nanmean(g128[:,t-1]):.2f}" for t in TIERS))

    # ---- sharpening vs capability, per recipe x region (trained / out-of-region) ----
    p("\n" + "=" * 78)
    p("SHARPENING vs CAPABILITY  (paired; contrast = SG-CG: >0 sharpening, <0 capability)")
    p("=" * 78)
    for name, (rl, trained) in rls.items():
        reg = {"trained": ([], []), "out": ([], [])}
        for (dd, tt) in base:
            if (dd, tt) not in rl:
                continue
            sg = list(paired_diff(base, rl, dd, tt, 0)); cg = list(paired_diff(base, rl, dd, tt, 1))
            k = "trained" if trained(dd, tt) else "out"
            reg[k][0].extend(sg); reg[k][1].extend(cg)
        p(f"\n-- {name} --")
        for k in ("trained", "out"):
            s, n = _summ(reg[k][0], reg[k][1])
            p(f"   {k:>8} (n={n:4d}): {s}")

    # ---- off-diagonal transfer, quantified ----
    p("\n" + "=" * 78)
    p("OFF-DIAGONAL TRANSFER  (CG: in-region vs out-region; peak-CG cell & its location)")
    p("=" * 78)
    p(f"   {'recipe':>8}  {'CG_in':>14}  {'CG_out':>14}  transfer  peak-CG cell")
    for name, (rl, trained) in rls.items():
        cin, cout = [], []
        cellmean = {}
        for (dd, tt) in base:
            if (dd, tt) not in rl:
                continue
            cg = paired_diff(base, rl, dd, tt, 1)
            cellmean[(dd, tt)] = cg.mean()
            (cin if trained(dd, tt) else cout).extend(list(cg))
        mi, ti = (np.mean(cin), tval(cin)) if cin else (0, 0)
        mo, to = (np.mean(cout), tval(cout)) if cout else (0, 0)
        pk = max(cellmean, key=cellmean.get)
        loc = "IN-region" if trained(*pk) else "OUT-region"
        # transfer index: out-region gain as a share of (in+out) positive gain
        tr = mo / (mi + mo) if (mi + mo) > 0 else float('nan')
        p(f"   {name:>8}  {mi:+.3f}(t{ti:+.1f})  {mo:+.3f}(t{to:+.1f})  "
          f"{tr:5.2f}     D{pk[0]}xT{pk[1]} {cellmean[pk]:+.3f} [{loc}]")
    p("\n   transfer index = CG_out / (CG_in + CG_out); ~0.5 means gains split evenly in/out,")
    p("   >0.5 means most capability gain lands OUTSIDE the trained region (= transfer).")
    out.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    base = load(BASE)
    rls = {name: (load(path), pred) for name, (path, pred) in RECIPES.items()}
    fig_base(base)
    fig_rl_grid(base, rls)
    fig_best_recipe(base, rls)
    tables(base, rls)
    print(f"wrote {OUT}/fig_base_capability.png, fig_rl_grid.png, fig_best_recipe.png, campaign_tables.txt")
