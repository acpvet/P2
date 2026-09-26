"""
P2 - Stage 7: the manuscript figure set.

Five figures, each written as both a 600-dpi PNG and a vector PDF. Nothing is
recomputed from memory: every number is read from the stage outputs, and the
two derived quantities this stage needs - which protein family drives each
discordant pair, and the negative-control distribution - are computed here
from those files rather than transcribed.

  Fig 1  workflow of the whole pipeline
  Fig 2  the linkage: scatter, effect sizes with leave-one-allergen-out range,
         and ROC for discriminating co-sensitised pairs
  Fig 3  the co-sensitization network, edges coloured by concordance
  Fig 4  discordance quadrants and the family that drives each discordant pair
  Fig 5  validation: observed vs expected co-occurrence, the negative control,
         and set-size bias before and after adjustment

Palette: Okabe-Ito, ordered so no adjacent pair falls below the colour-vision
separation floor. Every series is also directly labelled, so identity is never
carried by colour alone.

Run:
    python p2_step7_figures.py

Requires: pandas, numpy, scipy, matplotlib, networkx.
    pip install networkx
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr, rankdata

# --------------------------------------------------------------------------
PAIRS = "p2_esm2_pairs.csv"                 # Stage 6: clinical + all metrics
SETS = "p2_allergen_sets_final.csv"
PROTEIN_METRICS = "p2_protein_pair_metrics.csv"

CLINICAL = "z_adj"
PRIMARY = "z_esm2_cos_max"                  # best continuous metric
FAMILY_Z = "z_same_family_max"
# A pair counts as co-sensitised when its permutation p survives
# Benjamini-Hochberg at this level. This replaces an arbitrary z cut: the
# discarded z >= 2 edges were low-count pairs (n11 = 2-3) whose z is inflated
# by a near-zero null SD while their permutation p is 0.06-0.10.
Q_CUT = 0.10
Q_COL = "fdr_q_adj"
OUTDIR = "figures"
SEED = 20260925

BLUE, VERM, GREEN, AMBER, PINK = ("#0072B2", "#D55E00", "#009E73",
                                  "#E69F00", "#CC79A7")
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"

METRIC_LABELS = {
    "z_same_family_max": "Protein family shared",
    "z_esm2_cos_max": "ESM-2 embedding cosine",
    "z_esm2_cos_centered_max": "ESM-2 cosine (centred)",
    "z_window80_max": "FAO/WHO 80-aa window",
    "z_kmer6_max": "Identical 6-mers",
    "z_kmer8_max": "Identical 8-mers",
    "z_identity_global_max": "Global sequence identity",
}
# --------------------------------------------------------------------------

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {OUTDIR}/{name}.pdf and .png")


def tidy(ax, grid_axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        B = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ beta
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def roc(score, label):
    o = np.argsort(-score)
    y = label[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


# ============================== FIGURE 1 ==================================
def figure1():
    steps = [
        ("405 patients x 34 allergens\nsIgE class, complete grid",
         "Clinical input", BLUE),
        ("Curveball randomisation\npatient atopy and allergen\nprevalence held fixed",
         "Adjustment", BLUE),
        ("z_adj\nexcess co-sensitization\n525 allergen pairs", "Clinical layer", BLUE),
        ("UniProt tiered retrieval\nkeyword, WHO/IUIS name,\npan-allergen family",
         "Sequence input", GREEN),
        ("One representative per\nallergenic family\n155 proteins, 33 entries",
         "Curation", GREEN),
        ("Identity, FAO/WHO window,\nk-mers, family, ESM-2\n11 935 protein pairs",
         "Metrics", GREEN),
        ("Relabelling null\nset sizes held fixed\nz_metric", "Adjustment", GREEN),
        ("Spearman(z_metric, z_adj)\nnegative control\nleave-one-allergen-out",
         "Linkage", VERM),
        ("Concordant and discordant\npairs; driving family",
         "Interpretation", VERM),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    pos = {0: (1.55, 8.6), 1: (1.55, 6.3), 2: (1.55, 4.0),
           3: (5.4, 8.6), 4: (5.4, 6.3), 5: (5.4, 4.0), 6: (5.4, 1.7),
           7: (8.45, 2.9), 8: (8.45, 0.8)}
    w, h = 2.7, 1.5
    for i, (text, tag, col) in enumerate(steps):
        x, y = pos[i]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=1.1, edgecolor=col,
                                    facecolor=col + "18", zorder=2))
        ax.text(x, y + h / 2 - 0.17, tag, ha="center", va="center",
                fontsize=7, color=col, fontweight="bold", zorder=3)
        ax.text(x, y - 0.12, text, ha="center", va="center",
                fontsize=7.2, color=INK, zorder=3, linespacing=1.35)

    def arrow(a, b, col, rad=0.0):
        x1, y1 = pos[a]; x2, y2 = pos[b]
        ax.add_patch(FancyArrowPatch((x1, y1 - h / 2 - 0.04),
                                     (x2, y2 + h / 2 + 0.04),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.0, color=col,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1)
                     if abs(x1 - x2) < 0.1 else
                     FancyArrowPatch((x1 + w / 2 + 0.04, y1),
                                     (x2 - w / 2 - 0.04, y2),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.0, color=col,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))
    for a, b in [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)]:
        arrow(a, b, BLUE if a < 3 else GREEN)
    # clinical layer feeds the linkage: leave downward, then run horizontally
    # below the computational column so no arrow crosses a box
    ax.add_patch(FancyArrowPatch((1.55, 4.0 - h / 2 - 0.04),
                                 (8.45 - w / 2 - 0.04, 2.9),
                                 arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=VERM,
                                 connectionstyle="angle,angleA=-90,angleB=180,rad=10",
                                 zorder=1))
    ax.add_patch(FancyArrowPatch((5.4 + w / 2 + 0.04, 1.7),
                                 (8.45 - w / 2 - 0.04, 2.9 - 0.35),
                                 arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=VERM,
                                 connectionstyle="arc3,rad=-0.25", zorder=1))
    ax.add_patch(FancyArrowPatch((8.45, 2.9 - h / 2 - 0.04),
                                 (8.45, 0.8 + h / 2 + 0.04),
                                 arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=VERM, zorder=1))
    ax.text(1.55, 9.75, "Clinical", ha="center", fontsize=8.5,
            fontweight="bold", color=BLUE)
    ax.text(5.4, 9.75, "Computational", ha="center", fontsize=8.5,
            fontweight="bold", color=GREEN)
    ax.text(8.45, 9.75, "Linkage", ha="center", fontsize=8.5,
            fontweight="bold", color=VERM)
    save(fig, "fig1_workflow")


# ============================== FIGURE 2 ==================================
def figure2(d, metrics):
    y = d[CLINICAL].values
    fig = plt.figure(figsize=(7.2, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.55,
                          wspace=0.3)

    # A - scatter
    ax = fig.add_subplot(gs[0, 0]); tidy(ax)
    x = d[PRIMARY].values
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=7, color=BLUE, alpha=0.32, linewidths=0, zorder=2)
    bins = np.quantile(x[ok], np.linspace(0, 1, 9))
    cx, cy = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (x >= lo) & (x <= hi)
        if m.sum() > 4:
            cx.append(np.median(x[m])); cy.append(np.median(y[m]))
    ax.plot(cx, cy, color=VERM, linewidth=2, marker="o", markersize=4, zorder=3)
    r = spearmanr(x[ok], y[ok])
    ax.text(0.03, 0.96, f"$\\rho$ = {r.statistic:.3f}\n$p$ = {r.pvalue:.1e}",
            transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
    ax.text(0.97, 0.04, "binned median", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=VERM, fontweight="bold")
    ax.set_xlabel("ESM-2 embedding similarity (z)")
    ax.set_ylabel("Excess co-sensitization (z)")
    ax.set_title("A   Linkage", loc="left")

    # B - ROC
    ax = fig.add_subplot(gs[0, 1]); tidy(ax)
    lab = (d[Q_COL] < Q_CUT).astype(int).values
    cols = [BLUE, VERM, GREEN, AMBER, PINK]
    order = metrics.sort_values("rho", ascending=False).metric.tolist()[:4]
    for c, col in zip(order, cols):
        ok = np.isfinite(d[c])
        fpr, tpr, auc = roc(d.loc[ok, c].values, lab[ok.values])
        ax.plot(fpr, tpr, color=col, linewidth=1.6, zorder=3,
                label=f"{METRIC_LABELS.get(c, c)}  {auc:.2f}")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)),
            zorder=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"B   Co-sensitised pairs, FDR < {Q_CUT:g} (n = {int(lab.sum())})",
                 loc="left")
    ax.legend(loc="lower right", fontsize=6.3, handlelength=1.3,
              borderpad=0.2, labelspacing=0.3)

    # C - forest, full width
    ax = fig.add_subplot(gs[1, :]); tidy(ax, grid_axis="x")
    m = metrics.sort_values("rho")
    ypos = np.arange(len(m))
    for i, row in enumerate(m.itertuples()):
        ax.plot([row.loao_min, row.loao_max], [i, i], color=MUTED,
                linewidth=1.5, solid_capstyle="round", zorder=2)
    ax.scatter(m.rho, ypos, s=30, color=BLUE, zorder=3)
    for i, row in enumerate(m.itertuples()):
        ax.annotate(f"{row.rho:.3f}", (row.loao_max, i), fontsize=6.8,
                    color=INK, va="center", xytext=(5, 0),
                    textcoords="offset points")
    ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([METRIC_LABELS.get(v, v) for v in m.metric], fontsize=7.5)
    ax.set_xlim(-0.02, max(m.loao_max) * 1.18)
    ax.set_xlabel("Spearman $\\rho$ with excess co-sensitization "
                  "(line: leave-one-allergen-out range over 33 drops)")
    ax.set_title("C   Effect size and stability", loc="left")
    save(fig, "fig2_linkage")


# ============================== FIGURE 3 ==================================
def figure3(d):
    import networkx as nx
    sub = d[d[Q_COL] < Q_CUT]
    prev = {}
    for r in d.itertuples():
        prev[r.allergen_a.strip()] = r.n_pos_a
        prev[r.allergen_b.strip()] = r.n_pos_b
    G = nx.Graph()
    for r in sub.itertuples():
        G.add_edge(r.allergen_a.strip(), r.allergen_b.strip(),
                   z=r.z_adj, fam=int(r.same_family_any))

    # Components are laid out separately and packed, otherwise a detached
    # two-node component pushes the main graph into a single squashed blob.
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = {}
    main = G.subgraph(comps[0])
    p = nx.kamada_kawai_layout(main)
    xy = np.array(list(p.values()))
    span = np.ptp(xy, axis=0).max()
    for n, v in p.items():
        pos[n] = (np.array(v) - xy.mean(axis=0)) / span
    slot = -0.80
    for comp in comps[1:]:
        sg = G.subgraph(comp)
        if len(comp) > 2:
            q = {n: np.array(v) * 0.24
                 for n, v in nx.circular_layout(sg).items()}
        else:
            q = {n: np.array([i * 0.26 - 0.13, 0.0])
                 for i, n in enumerate(sorted(comp))}
        for n, v in q.items():
            pos[n] = v + np.array([slot, -0.66])
        slot += 0.78

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.axis("off"); ax.set_aspect("equal")
    for u, v, a in G.edges(data=True):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=GREEN if a["fam"] else VERM,
                linewidth=0.6 + 0.85 * max(a["z"] - 2.0, 0.1),
                alpha=0.7, zorder=1, solid_capstyle="round")
    sizes = [24 + 8 * prev.get(n, 1) ** 0.62 for n in G.nodes]
    ax.scatter([pos[n][0] for n in G.nodes], [pos[n][1] for n in G.nodes],
               s=sizes, color="white", edgecolors=BLUE, linewidths=1.2, zorder=2)
    # Labels are placed by trying four positions around the node and keeping
    # the first whose text box clears every box already placed. Without this
    # two adjacent nodes both push their label to the same side and overlap.
    CH, PAD = 0.030, 0.006          # half text height, gap, in data units
    # node markers are obstacles too, so a label never sits on a circle
    NR = 0.026
    boxes = [(pos[n][0] - NR, pos[n][1] - NR, pos[n][0] + NR, pos[n][1] + NR)
             for n in G.nodes]

    def half_width(t):
        return 0.0125 * len(t)

    def box(ax_, ay_, hw, ha_, va_):
        x0 = ax_ - (hw if ha_ == "center" else (2 * hw if ha_ == "right" else 0))
        y0 = ay_ - (CH if va_ == "center" else (2 * CH if va_ == "top" else 0))
        return (x0 - PAD, y0 - PAD, x0 + 2 * hw + PAD, y0 + 2 * CH + PAD)

    def clashes(b):
        return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3])
                   for o in boxes)

    for n in sorted(G.nodes, key=lambda m: (-pos[m][1], pos[m][0])):
        px, py = pos[n]
        text = n.split(" (")[0]
        hw = half_width(text)
        options = [((px, py + 0.045), "center", "bottom", (0, 5.5)),
                   ((px, py - 0.045), "center", "top", (0, -5.5)),
                   ((px + 0.035, py), "left", "center", (7, 0)),
                   ((px - 0.035, py), "right", "center", (-7, 0))]
        chosen = options[0]
        for anchor, ha_, va_, off in options:
            if not clashes(box(anchor[0], anchor[1], hw, ha_, va_)):
                chosen = (anchor, ha_, va_, off)
                break
        anchor, ha_, va_, off = chosen
        boxes.append(box(anchor[0], anchor[1], hw, ha_, va_))
        ax.annotate(text, (px, py), fontsize=6.4, ha=ha_, va=va_,
                    zorder=3, color=INK, xytext=off, textcoords="offset points")

    ax.plot([], [], color=GREEN, linewidth=2, label="shares a protein family")
    ax.plot([], [], color=VERM, linewidth=2, label="no shared family")
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 1.02))
    ax.margins(0.12)
    ax.set_title(f"Co-sensitization beyond overall atopy\n"
                 f"edges: FDR < {Q_CUT:g}   ·   "
                 f"node size: sensitized patients   ·   "
                 f"edge width: strength", loc="left", fontsize=8.5)
    save(fig, "fig3_network")


# ============================== FIGURE 4 ==================================
def driving_family(d, sets, pp):
    fam = dict(zip(sets.accession, sets.families.fillna("unassigned")))
    mem = sets.groupby("panel_entry").accession.apply(
        lambda s: list(dict.fromkeys(s))).to_dict()
    lk = {}
    for r in pp.itertuples():
        lk[(r.acc_a, r.acc_b)] = r.identity_global
        lk[(r.acc_b, r.acc_a)] = r.identity_global
    out = []
    for r in d.itertuples():
        best = (-1, None)
        for x in mem[r.allergen_a.strip()]:
            for y in mem[r.allergen_b.strip()]:
                if x == y:
                    continue
                v = lk[(x, y)]
                if v > best[0]:
                    best = (v, x)
        f = fam[best[1]].split(";")[0].split(",")[0].replace(" family", "").strip()
        out.append(f if f else "unassigned")
    return out


def figure4(d, sets, pp):
    x = d[PRIMARY].values
    y = d[CLINICAL].values
    hi_x, lo_x = np.nanpercentile(x, 80), np.nanpercentile(x, 20)
    hi_y, lo_y = np.nanpercentile(y, 80), np.nanpercentile(y, 20)
    quad = np.select(
        [(x >= hi_x) & (y >= hi_y), (x >= hi_x) & (y <= lo_y),
         (x <= lo_x) & (y >= hi_y), (x <= lo_x) & (y <= lo_y)],
        ["concordant high", "similar, not co-sensitised",
         "co-sensitised, not similar", "concordant low"], default="middle")
    d = d.assign(quadrant=quad)

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0]); tidy(ax)
    ax.axvspan(hi_x, np.nanmax(x) * 1.05, color=GRID, alpha=0.35, zorder=0)
    ax.axhspan(hi_y, np.nanmax(y) * 1.1, color=GRID, alpha=0.35, zorder=0)
    ax.scatter(x, y, s=7, color=MUTED, alpha=0.25, linewidths=0, zorder=2)
    for q, col in (("similar, not co-sensitised", VERM),
                   ("co-sensitised, not similar", BLUE),
                   ("concordant high", GREEN)):
        m = d.quadrant == q
        ax.scatter(x[m.values], y[m.values], s=18, color=col, zorder=3,
                   linewidths=0, label=f"{q} ({int(m.sum())})")
    lab = d[d.quadrant == "similar, not co-sensitised"].nsmallest(5, CLINICAL)
    xmax = np.nanmax(x)
    ax.set_xlim(np.nanmin(x) - 0.4, xmax + 3.0)
    ys = np.linspace(-1.2, -4.2, len(lab))
    for (r, ly) in zip(lab.itertuples(), ys):
        px, py = getattr(r, PRIMARY), getattr(r, CLINICAL)
        ax.annotate(f"{r.allergen_a}–{r.allergen_b}", (px, py),
                    xytext=(xmax + 0.35, ly), textcoords="data",
                    fontsize=6.3, color=VERM, va="center", ha="left",
                    zorder=5,
                    arrowprops=dict(arrowstyle="-", color=VERM, linewidth=0.5,
                                    shrinkA=0, shrinkB=2, alpha=0.8))
    ax.axhline(0, color=INK, linewidth=0.7, zorder=1)
    ax.set_xlabel("ESM-2 embedding similarity (z)")
    ax.set_ylabel("Excess co-sensitization (z)")
    ax.set_title("A   Concordance and discordance", loc="left")
    ax.legend(loc="lower left", fontsize=6.6)

    ax = fig.add_subplot(gs[1]); tidy(ax, grid_axis="x")
    sub = d[d.quadrant == "similar, not co-sensitised"].copy()
    sub["family"] = driving_family(sub, sets, pp)
    cnt = sub.family.value_counts()
    ypos = np.arange(len(cnt))[::-1]
    ax.barh(ypos, cnt.values, color=VERM, height=0.6, zorder=2)
    for i, (f, v) in zip(ypos, cnt.items()):
        ax.text(v + 0.12, i, str(v), va="center", fontsize=7.5, color=INK)
    ax.set_yticks(ypos); ax.set_yticklabels(cnt.index, fontsize=7.2)
    ax.set_xlabel("Discordant pairs (similar, not co-sensitised)")
    ax.set_xlim(0, cnt.max() * 1.22)
    ax.set_title("B   Protein family driving the similarity", loc="left")
    save(fig, "fig4_discordance")
    return sub


# ============================== FIGURE 5 ==================================
def figure5(d, metrics):
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], wspace=0.75,
                          hspace=0.62)

    # A - observed vs expected co-occurrence
    ax = fig.add_subplot(gs[:, 0]); tidy(ax, grid_axis="x")
    top = d.nlargest(18, CLINICAL).iloc[::-1]
    ypos = np.arange(len(top))
    ax.errorbar(top.n11_null_mean, ypos, xerr=top.n11_null_sd, fmt="o",
                color=MUTED, markersize=3.2, elinewidth=1.1, capsize=1.8,
                zorder=2)
    ax.scatter(top.n11, ypos, s=24, color=VERM, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{a}–{b}" for a, b in
                        zip(top.allergen_a, top.allergen_b)], fontsize=6.2)
    ax.set_xlabel("Patients positive to both")
    ax.text(0.97, 0.05, "observed", transform=ax.transAxes, ha="right",
            fontsize=7, color=VERM, fontweight="bold")
    ax.text(0.97, 0.13, "expected ± SD", transform=ax.transAxes, ha="right",
            fontsize=7, color=MUTED, fontweight="bold")
    ax.set_title("A   Excess over the null", loc="left")

    # B - negative control
    ax = fig.add_subplot(gs[0, 1]); tidy(ax)
    y = d[CLINICAL].values
    ok = np.isfinite(d[PRIMARY])
    xb, yb = d.loc[ok, PRIMARY].values, y[ok.values]
    rng = np.random.default_rng(SEED + 1)
    null = np.array([spearmanr(xb, rng.permutation(yb)).statistic
                     for _ in range(1000)])
    obs = spearmanr(xb, yb).statistic
    ax.hist(null, bins=34, color=GRID, edgecolor=MUTED, linewidth=0.4, zorder=2)
    ax.axvline(obs, color=VERM, linewidth=1.8, zorder=3)
    ax.annotate("observed", (obs, ax.get_ylim()[1] * 0.92), fontsize=7,
                color=VERM, ha="right", xytext=(-4, 0),
                textcoords="offset points", fontweight="bold")
    ax.set_xlabel("Spearman $\\rho$"); ax.set_ylabel("Shuffles")
    ax.set_title("B   Clinical labels shuffled (1000×)", loc="left")

    # C - set-size bias before and after
    ax = fig.add_subplot(gs[1, 1]); tidy(ax, grid_axis="x")
    raw = {"z_identity_global_max": "identity_global_max",
           "z_window80_max": "window80_max", "z_kmer6_max": "kmer6_max",
           "z_kmer8_max": "kmer8_max", "z_same_family_max": "same_family_any"}
    names, before, after = [], [], []
    for zc, rc in raw.items():
        if rc not in d.columns:
            continue
        names.append(METRIC_LABELS.get(zc, zc))
        before.append(spearmanr(d[rc], d.n_combinations).statistic)
        after.append(spearmanr(d[zc], d.n_combinations,
                               nan_policy="omit").statistic)
    ypos = np.arange(len(names))[::-1]
    for i, b, a in zip(ypos, before, after):
        ax.plot([b, a], [i, i], color=GRID, linewidth=1.6, zorder=1)
    ax.scatter(before, ypos, s=24, color=AMBER, zorder=3)
    ax.scatter(after, ypos, s=24, color=BLUE, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("$\\rho$ with protein combinations")
    ax.text(0.02, 0.06, "raw", transform=ax.transAxes, fontsize=7,
            color=AMBER, fontweight="bold")
    ax.text(0.02, 0.15, "adjusted", transform=ax.transAxes, fontsize=7,
            color=BLUE, fontweight="bold")
    ax.set_title("C   Set-size bias", loc="left")
    save(fig, "fig5_validation")


def main():
    print("=" * 74)
    print("P2 - STAGE 7: FIGURES")
    print("=" * 74)
    d = pd.read_csv(PAIRS)
    sets = pd.read_csv(SETS)
    pp = pd.read_csv(PROTEIN_METRICS)
    d["allergen_a"] = d.allergen_a.str.strip()
    d["allergen_b"] = d.allergen_b.str.strip()
    assert CLINICAL in d and PRIMARY in d, "expected Stage 6 output"
    print(f"\nPairs {len(d)}   proteins {sets.accession.nunique()}   "
          f"panel entries {sets.panel_entry.nunique()}")

    y = d[CLINICAL].values
    rows = []
    entries = sorted(set(d.allergen_a) | set(d.allergen_b))
    for c in METRIC_LABELS:
        if c not in d.columns:
            continue
        ok = np.isfinite(d[c])
        r = spearmanr(d.loc[ok, c], y[ok.values])
        j = []
        for e in entries:
            k = ok & ~((d.allergen_a == e) | (d.allergen_b == e))
            if k.sum() > 30:
                j.append(spearmanr(d.loc[k, c], y[k.values]).statistic)
        okf = ok & np.isfinite(d[FAMILY_Z])
        rows.append({"metric": c, "rho": r.statistic, "p": r.pvalue,
                     "loao_min": min(j), "loao_max": max(j),
                     "partial_family": partial_spearman(
                         d.loc[okf, c].values, y[okf.values],
                         d.loc[okf, FAMILY_Z].values)
                     if c != FAMILY_Z else np.nan})
    metrics = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    metrics.to_csv(os.path.join(OUTDIR, "table1_metrics.csv"), index=False)
    print("\n=== TABLE 1 ===")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\nDrawing:")
    figure1()
    figure2(d, metrics)
    figure3(d)
    disc = figure4(d, sets, pp)
    figure5(d, metrics)

    disc.to_csv(os.path.join(OUTDIR, "table2_discordant.csv"), index=False)
    print("\n=== DRIVING FAMILY, DISCORDANT PAIRS ===")
    print(disc.family.value_counts().to_string())
    print(f"\nAll files in {OUTDIR}/  (PDF vector + 600-dpi PNG)")
    print("=" * 74)


if __name__ == "__main__":
    main()
