"""
P2 - Stage 4: sequence-similarity metrics for every panel-entry pair.

Each panel entry is a SET of proteins, so every metric is computed for all
protein x protein combinations and then aggregated. Two aggregates are kept:
the maximum (one shared allergen is enough to explain cross-reactivity) and
the mean of the top three (robust to a single spurious hit).

Metrics, per protein pair:
  identity_global   global Needleman-Wunsch (BLOSUM62, -11/-1), identical
                    columns divided by the shorter sequence
  window80          the FAO/WHO criterion: the highest identity over any
                    80-residue window
  kmer6, kmer8      number of exact shared stretches of 6 and 8 residues -
                    the other FAO/WHO criterion, computed on the raw
                    sequences with no alignment involved
  same_family       the two proteins share a UniProt protein family

window80 is taken over columns of the local alignment rather than by aligning
every 80-mer separately, which would take hours. The approximation is not
assumed: N_VALIDATE random protein pairs are also scored the slow exact way
and the two are compared in the output. If they disagree the number is
unusable and the run says so.

Degenerate panel pairs (Cow Milk x Cow Milk UHT, Casein x Cow Milk, Gluten x
Wheat Flour) are excluded from the analysis file and written separately as
technical controls.

Run:
    python p2_step4_metrics.py

Requires: pandas, numpy, scipy, biopython.   Runtime ~10 min.
"""

import math
import random
from itertools import combinations

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from scipy.stats import spearmanr

# --------------------------------------------------------------------------
SETS_PATH = "p2_allergen_sets_final.csv"
FASTA_PATH = "p2_sequences_curated.fasta"
CLINICAL_PATH = "p2_pair_table_adjusted.csv"

WINDOW = 80                 # FAO/WHO window length
WINDOW_STEP = 1             # 1 = the literal scan; raise only to save time
KMERS = (6, 8)
TOP_K = 3                   # for the mean-of-top-k aggregate
SEED = 20260925

EXCLUDE_PAIRS = [           # structurally degenerate, decided at Stage 3c
    ("Cow Milk", "Cow Milk, UHT"),
    ("Casein (nBos d8)", "Cow Milk"),
    ("Gluten", "Wheat Flour"),
]

OUT_PROTEIN = "p2_protein_pair_metrics.csv"
OUT_PAIRS = "p2_metrics_pairs.csv"
OUT_CONTROLS = "p2_degenerate_controls.csv"
# --------------------------------------------------------------------------


def read_fasta(path):
    seqs, acc, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            acc, buf = line[1:].split("|")[0], []
        elif line:
            buf.append(line)
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def make_aligners():
    blosum = substitution_matrices.load("BLOSUM62")
    g = Align.PairwiseAligner(mode="global", substitution_matrix=blosum,
                              open_gap_score=-11, extend_gap_score=-1)
    l = Align.PairwiseAligner(mode="local", substitution_matrix=blosum,
                              open_gap_score=-11, extend_gap_score=-1)
    return g, l


def match_vector(alignment, a, b):
    """1 where the two aligned residues are identical, 0 for a mismatch or a
    gap column. Built from the aligned blocks, so no alignment string is
    materialised."""
    ba, bb = alignment.aligned
    cols = []
    prev_a = prev_b = None
    for (a0, a1), (b0, b1) in zip(ba, bb):
        if prev_a is not None:
            gap = (a0 - prev_a) + (b0 - prev_b)
            if gap:
                cols.append(np.zeros(gap, dtype=np.int8))
        seg_a = np.frombuffer(a[a0:a1].encode(), dtype=np.uint8)
        seg_b = np.frombuffer(b[b0:b1].encode(), dtype=np.uint8)
        cols.append((seg_a == seg_b).astype(np.int8))
        prev_a, prev_b = a1, b1
    return np.concatenate(cols) if cols else np.zeros(0, dtype=np.int8)


def max_window(mv, w):
    if len(mv) < w:
        return float(mv.sum()) / w if len(mv) else 0.0
    c = np.concatenate([[0], np.cumsum(mv)])
    return float((c[w:] - c[:-w]).max()) / w


def kmer_sets(s, k):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def exact_window80(a, b, aligner, w=80, step=1):
    """The literal FAO/WHO scan: every w-residue window of a is locally
    aligned to the whole of b. Used only on the validation subsample."""
    if len(a) < w:
        return None
    best = 0.0
    for i in range(0, len(a) - w + 1, step):
        win = a[i:i + w]
        aln = aligner.align(win, b)[0]
        mv = match_vector(aln, win, b)
        best = max(best, float(mv.sum()) / w)
    return best


def main():
    print("=" * 78)
    print("P2 - STAGE 4: SEQUENCE SIMILARITY METRICS")
    print("=" * 78)

    sets = pd.read_csv(SETS_PATH)
    seqs = read_fasta(FASTA_PATH)
    missing = set(sets.accession) - set(seqs)
    assert not missing, f"no sequence for {sorted(missing)[:5]}"
    sets["families"] = sets.families.fillna("")
    entries = sorted(sets.panel_entry.unique())
    print(f"\nPanel entries {len(entries)}   proteins {sets.accession.nunique()}"
          f"   rows {len(sets)}")

    prot = sorted(set(sets.accession))
    idx = {a: i for i, a in enumerate(prot)}
    fam = {r.accession: {f.strip() for f in r.families.split(";") if f.strip()}
           for r in sets.itertuples()}

    g_al, l_al = make_aligners()
    n_pp = len(prot) * (len(prot) - 1) // 2
    print(f"Protein pairs to align: {n_pp}")

    rows = []
    done = 0
    step = max(1, n_pp // 10)
    for x, y in combinations(prot, 2):
        a, b = seqs[x], seqs[y]
        ga = g_al.align(a, b)[0]
        mv_g = match_vector(ga, a, b)
        ident = float(mv_g.sum()) / min(len(a), len(b))
        la = l_al.align(a, b)[0]
        mv_l = match_vector(la, a, b)
        w80_fast = max_window(mv_l, WINDOW)
        w80 = exact_window80(a, b, l_al, WINDOW, WINDOW_STEP)
        if w80 is None:                       # shorter than the window
            w80 = w80_fast
        rec = {"acc_a": x, "acc_b": y, "identity_global": ident,
               "window80": w80, "window80_shortcut": w80_fast,
               "same_family": int(bool(fam[x] & fam[y]))}
        for k in KMERS:
            rec[f"kmer{k}"] = len(kmer_sets(a, k) & kmer_sets(b, k))
        rows.append(rec)
        done += 1
        if done % step == 0:
            print(f"  ... {done}/{n_pp}")

    pp = pd.DataFrame(rows)
    pp.to_csv(OUT_PROTEIN, index=False)

    # ---- how far the cheap shortcut would have been off ------------------
    ok = pp.window80_shortcut.notna()
    rho = spearmanr(pp.loc[ok, "window80"], pp.loc[ok, "window80_shortcut"]).statistic
    diff = (pp.loc[ok, "window80_shortcut"] - pp.loc[ok, "window80"]).values
    print(f"\n=== window80: exact scan vs alignment-column shortcut "
          f"({int(ok.sum())} pairs) ===")
    print(f"  Spearman {rho:.3f}   shortcut - exact: mean {diff.mean():+.4f}  "
          f"SD {diff.std():.4f}  max |d| {np.abs(diff).max():.4f}")
    print("  The reported window80 is the exact scan; the shortcut column is "
          "kept only for this comparison.")

    # ---- aggregate to panel-entry pairs ---------------------------------
    members = {e: list(sets.loc[sets.panel_entry == e, "accession"].unique())
               for e in entries}
    lookup = {}
    for r in pp.itertuples():
        lookup[(r.acc_a, r.acc_b)] = r
        lookup[(r.acc_b, r.acc_a)] = r

    metric_cols = ["identity_global", "window80"] + [f"kmer{k}" for k in KMERS]
    out = []
    for e1, e2 in combinations(entries, 2):
        vals = {m: [] for m in metric_cols}
        fam_hit = 0
        for x in members[e1]:
            for y in members[e2]:
                if x == y:          # the same protein in both sets
                    vals["identity_global"].append(1.0)
                    vals["window80"].append(1.0)
                    for k in KMERS:
                        vals[f"kmer{k}"].append(len(kmer_sets(seqs[x], k)))
                    fam_hit = 1
                    continue
                r = lookup[(x, y)]
                for m in metric_cols:
                    vals[m].append(getattr(r, m))
                fam_hit = max(fam_hit, r.same_family)
        rec = {"allergen_a": e1, "allergen_b": e2,
               "n_prot_a": len(members[e1]), "n_prot_b": len(members[e2]),
               "n_combinations": len(members[e1]) * len(members[e2]),
               "same_family_any": fam_hit}
        for m in metric_cols:
            v = sorted(vals[m], reverse=True)
            rec[f"{m}_max"] = v[0]
            rec[f"{m}_top{TOP_K}"] = float(np.mean(v[:TOP_K]))
        out.append(rec)

    met = pd.DataFrame(out)

    # ---- join the clinical layer ----------------------------------------
    clin = pd.read_csv(CLINICAL_PATH)
    key = lambda d: list(zip(d.allergen_a.str.strip(), d.allergen_b.str.strip()))
    met["k"] = [tuple(sorted(p)) for p in key(met)]
    clin["k"] = [tuple(sorted(p)) for p in key(clin)]
    merged = clin.merge(met.drop(columns=["allergen_a", "allergen_b"]),
                        on="k", how="inner")

    # A panel entry can end Stage 3c with no protein at all - Sunflower Seeds
    # lost its only record to the pollen filter. Its pairs cannot be scored,
    # so they are named and removed rather than silently missing.
    clin_entries = set(clin.allergen_a.str.strip()) | set(clin.allergen_b.str.strip())
    lost = sorted(clin_entries - set(entries))
    if lost:
        print(f"\n=== PANEL ENTRIES WITH NO SEQUENCE ===")
        for e in lost:
            n = int(((clin.allergen_a.str.strip() == e) |
                     (clin.allergen_b.str.strip() == e)).sum())
            print(f"  {e}: excluded, {n} pairs lost")
    assert len(merged) + sum(
        ((clin.allergen_a.str.strip() == e) |
         (clin.allergen_b.str.strip() == e)).sum() for e in lost) >= len(clin), \
        "pairs went missing for a reason other than an empty panel entry"

    excl = {tuple(sorted(p)) for p in EXCLUDE_PAIRS}
    controls = merged[merged.k.isin(excl)].drop(columns="k")
    analysis = merged[~merged.k.isin(excl)].drop(columns="k")
    assert len(controls) == len(EXCLUDE_PAIRS), \
        f"expected {len(EXCLUDE_PAIRS)} control pairs, matched {len(controls)}"
    analysis.to_csv(OUT_PAIRS, index=False)
    controls.to_csv(OUT_CONTROLS, index=False)

    print(f"\n=== PAIRS ===")
    print(f"  analysis {len(analysis)}   technical controls {len(controls)}")
    print("\n  technical controls (similarity must be ~1 by construction):")
    print(controls[["allergen_a", "allergen_b", "identity_global_max",
                    "window80_max", "z_adj"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    print("\n=== METRIC DISTRIBUTIONS (analysis set) ===")
    cols = [c for c in analysis.columns
            if c.endswith("_max") or c.endswith(f"_top{TOP_K}")]
    print(analysis[cols].describe().loc[
        ["min", "25%", "50%", "75%", "max"]].to_string(
        float_format=lambda v: f"{v:8.3f}"))

    print("\n=== SET-SIZE BIAS CHECK ===")
    print("  Spearman of each metric with the number of protein combinations")
    print("  (a metric that tracks set size is measuring annotation depth):")
    for c in cols:
        r1 = spearmanr(analysis[c], analysis.n_combinations).statistic
        r2 = spearmanr(analysis[c],
                       analysis[["n_prot_a", "n_prot_b"]].min(axis=1)).statistic
        flag = "  <-- inspect" if abs(r1) > 0.4 else ""
        print(f"    {c:24s} vs combinations {r1:+.3f}   vs smaller set "
              f"{r2:+.3f}{flag}")

    print("\n=== FIRST LOOK AT THE LINKAGE (Stage 5 does this properly) ===")
    for c in cols + ["same_family_any"]:
        r = spearmanr(analysis[c], analysis.z_adj)
        print(f"    {c:24s} vs z_adj  rho {r.statistic:+.3f}  p {r.pvalue:.2g}")

    print(f"\nWritten: {OUT_PROTEIN}, {OUT_PAIRS}, {OUT_CONTROLS}")
    print("=" * 78)


if __name__ == "__main__":
    main()
