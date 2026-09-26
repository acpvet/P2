"""
P2 - Stage 5: link computational similarity to clinical co-sensitization.

Both sides of the comparison are now adjusted the same way.

Clinical side (Stage 2): z_adj, the excess co-occurrence of a pair over a null
that holds each patient's total number of sensitizations and each allergen's
prevalence fixed.

Computational side (here): the raw metrics from Stage 4 cannot be used as they
stand. identity_global_top3 correlates +0.60 with the number of protein
combinations in a pair, because a maximum over N draws grows with N and set
size reflects how deeply an organism happens to be annotated. So each metric
gets the same treatment as the clinical layer: the 155 proteins are randomly
relabelled across the panel entries with every set size held fixed, the
aggregate is recomputed, and the observed value is scored against that null.

    z_metric = (observed - mean_null) / sd_null

Relabelling is a single permutation of protein identities applied to the whole
membership table, so proteins shared between entries stay shared and no set
changes size. No sequence is realigned: the 155 x 155 metric matrices from
Stage 4 are indexed directly.

Validation, in the order it is reported:
  - negative control: shuffle z_adj against the metrics; every correlation
    must collapse to zero;
  - leave-one-allergen-out: drop all pairs containing one panel entry and
    recompute, for all 33 in turn. A result carried by a single allergen shows
    up here and nowhere else;
  - set-size check repeated on the z-scored metrics, which must now be flat.

Run:
    python p2_step5_linkage.py

Requires: pandas, numpy, scipy.   Runtime ~5 min.
"""

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

# --------------------------------------------------------------------------
SETS_PATH = "p2_allergen_sets_final.csv"
PROTEIN_METRICS = "p2_protein_pair_metrics.csv"
PAIRS_PATH = "p2_metrics_pairs.csv"
FASTA_PATH = "p2_sequences_curated.fasta"

METRICS = ["identity_global", "window80", "kmer6", "kmer8", "same_family"]
TOP_K = 3
N_PERM = 10000
SEED = 20260925
CLINICAL = "z_adj"

OUT_PAIRS = "p2_linkage_pairs.csv"
OUT_SUMMARY = "p2_linkage_summary.csv"
OUT_DISCORDANT = "p2_discordant_pairs.csv"
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


def build_matrices(pp, prot, seqs):
    """One symmetric matrix per metric, with the diagonal set to what a
    protein scores against itself."""
    idx = {a: i for i, a in enumerate(prot)}
    n = len(prot)
    M = np.zeros((len(METRICS), n, n), dtype=np.float64)
    ia = pp.acc_a.map(idx).values
    ib = pp.acc_b.map(idx).values
    for m, name in enumerate(METRICS):
        v = pp[name].values.astype(float)
        M[m, ia, ib] = v
        M[m, ib, ia] = v
    for i, a in enumerate(prot):
        s = seqs[a]
        for m, name in enumerate(METRICS):
            if name in ("identity_global", "window80", "same_family"):
                M[m, i, i] = 1.0
            else:
                k = int(name[4:])
                M[m, i, i] = len({s[j:j + k] for j in range(len(s) - k + 1)})
    return M


def aggregate(M, ia, ib):
    """max and mean-of-top-k over the submatrix, for every metric at once."""
    sub = M[:, ia][:, :, ib].reshape(len(METRICS), -1)
    mx = sub.max(axis=1)
    k = min(TOP_K, sub.shape[1])
    part = -np.partition(-sub, k - 1, axis=1)[:, :k]
    return mx, part.mean(axis=1)


def partial_spearman(x, y, z):
    """Spearman of x and y with z partialled out, on ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b, a, rcond=None)
        return a - b @ beta
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def main():
    print("=" * 78)
    print("P2 - STAGE 5: LINKING SIMILARITY TO CLINICAL CO-SENSITIZATION")
    print("=" * 78)

    sets = pd.read_csv(SETS_PATH)
    pp = pd.read_csv(PROTEIN_METRICS)
    pairs = pd.read_csv(PAIRS_PATH)
    seqs = read_fasta(FASTA_PATH)

    entries = sorted(sets.panel_entry.unique())
    prot = sorted(set(sets.accession))
    idx = {a: i for i, a in enumerate(prot)}
    members = {e: np.array([idx[a] for a in
                            sets.loc[sets.panel_entry == e, "accession"].unique()])
               for e in entries}
    print(f"\nPanel entries {len(entries)}   proteins {len(prot)}   "
          f"pairs in analysis {len(pairs)}")

    M = build_matrices(pp, prot, seqs)

    pair_keys = [(r.allergen_a.strip(), r.allergen_b.strip())
                 for r in pairs.itertuples()]
    ia_list = [members[a] for a, b in pair_keys]
    ib_list = [members[b] for a, b in pair_keys]

    obs_mx, obs_tk = [], []
    for ia, ib in zip(ia_list, ib_list):
        mx, tk = aggregate(M, ia, ib)
        obs_mx.append(mx)
        obs_tk.append(tk)
    obs_mx = np.array(obs_mx)
    obs_tk = np.array(obs_tk)

    # sanity: the observed aggregates must reproduce Stage 4
    for m, name in enumerate(METRICS):
        col = f"{name}_max" if name != "same_family" else "same_family_any"
        if col in pairs.columns:
            d = np.abs(obs_mx[:, m] - pairs[col].values).max()
            assert d < 1e-9, f"{col} does not reproduce Stage 4 (max diff {d})"
    print("Observed aggregates reproduce Stage 4 exactly.")

    # ---- set-size-preserving null ---------------------------------------
    rng = np.random.default_rng(SEED)
    n = len(prot)
    s1_mx = np.zeros_like(obs_mx); s2_mx = np.zeros_like(obs_mx)
    s1_tk = np.zeros_like(obs_tk); s2_tk = np.zeros_like(obs_tk)
    print(f"\nRelabelling null: {N_PERM} permutations of {n} proteins "
          f"(set sizes fixed), seed {SEED}")
    step = max(1, N_PERM // 10)
    for it in range(N_PERM):
        p = rng.permutation(n)
        for i, (ia, ib) in enumerate(zip(ia_list, ib_list)):
            mx, tk = aggregate(M, p[ia], p[ib])
            s1_mx[i] += mx; s2_mx[i] += mx ** 2
            s1_tk[i] += tk; s2_tk[i] += tk ** 2
        if (it + 1) % step == 0:
            print(f"  ... {it+1}/{N_PERM}")

    def zscore(obs, s1, s2):
        mu = s1 / N_PERM
        sd = np.sqrt(np.maximum(s2 / N_PERM - mu ** 2, 0.0))
        z = np.where(sd > 0, (obs - mu) / np.where(sd > 0, sd, 1.0), np.nan)
        return mu, sd, z

    out = pairs.copy()
    zcols = []
    for m, name in enumerate(METRICS):
        for tag, obs, s1, s2 in (("max", obs_mx[:, m], s1_mx[:, m], s2_mx[:, m]),
                                 (f"top{TOP_K}", obs_tk[:, m], s1_tk[:, m],
                                  s2_tk[:, m])):
            mu, sd, z = zscore(obs, s1, s2)
            out[f"z_{name}_{tag}"] = z
            out[f"null_{name}_{tag}_mean"] = mu
            zcols.append(f"z_{name}_{tag}")
    out.to_csv(OUT_PAIRS, index=False)

    y = out[CLINICAL].values

    print("\n=== SET-SIZE BIAS AFTER ADJUSTMENT ===")
    print("  (Stage 4 raw metrics ran +0.51 to +0.65 against set size)")
    for c in zcols:
        ok = np.isfinite(out[c])
        r = spearmanr(out.loc[ok, c], out.loc[ok, "n_combinations"]).statistic
        print(f"    {c:28s} vs combinations {r:+.3f}"
              f"{'   <-- still biased' if abs(r) > 0.3 else ''}")

    # ---- linkage ---------------------------------------------------------
    print(f"\n=== LINKAGE: metric vs {CLINICAL} ({len(out)} pairs) ===")
    rows = []
    for c in zcols:
        ok = np.isfinite(out[c])
        r = spearmanr(out.loc[ok, c], y[ok])
        pr = partial_spearman(out.loc[ok, c].values, y[ok],
                              out.loc[ok, "n_combinations"].values)
        rows.append({"metric": c, "n": int(ok.sum()), "rho": r.statistic,
                     "p": r.pvalue, "rho_partial_setsize": pr})
    summ = pd.DataFrame(rows).sort_values("rho", ascending=False)

    # leave-one-allergen-out
    jack = {c: [] for c in zcols}
    for e in entries:
        keep = ~((out.allergen_a.str.strip() == e) |
                 (out.allergen_b.str.strip() == e))
        for c in zcols:
            ok = keep & np.isfinite(out[c])
            if ok.sum() > 30:
                jack[c].append(spearmanr(out.loc[ok, c], y[ok.values]).statistic)
    summ["loao_min"] = summ.metric.map(lambda c: np.min(jack[c]))
    summ["loao_max"] = summ.metric.map(lambda c: np.max(jack[c]))
    summ["loao_worst_entry"] = summ.metric.map(
        lambda c: entries[int(np.argmin(jack[c]))])
    summ.to_csv(OUT_SUMMARY, index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # negative control
    print("\n=== NEGATIVE CONTROL (clinical labels shuffled, 1000x) ===")
    rng2 = np.random.default_rng(SEED + 1)
    best = summ.metric.iloc[0]
    okb = np.isfinite(out[best])
    xb = out.loc[okb, best].values
    yb = y[okb.values]
    nullr = [spearmanr(xb, rng2.permutation(yb)).statistic for _ in range(1000)]
    obs_r = summ.rho.iloc[0]
    print(f"  best metric {best}: observed rho {obs_r:+.3f}")
    print(f"  shuffled: mean {np.mean(nullr):+.4f}  SD {np.std(nullr):.4f}  "
          f"|rho| >= observed in {int(np.sum(np.abs(nullr) >= abs(obs_r)))}/1000")

    # ---- does anything add to family membership? -------------------------
    fam_col = "z_same_family_max"
    print(f"\n=== INCREMENTAL VALUE OVER {fam_col} ===")
    print("  Spearman with z_adj, with family membership partialled out.")
    print("  A sequence metric that only restates family membership goes to 0.")
    for c in zcols:
        if c == fam_col:
            continue
        ok = np.isfinite(out[c]) & np.isfinite(out[fam_col])
        pr = partial_spearman(out.loc[ok, c].values, y[ok],
                              out.loc[ok, fam_col].values)
        raw = spearmanr(out.loc[ok, c], y[ok]).statistic
        print(f"    {c:28s} rho {raw:+.3f}  ->  partial {pr:+.3f}")

    # ---- discordance -----------------------------------------------------
    print("\n=== DISCORDANCE ===")
    zc = out[best].values
    hi_c = np.nanpercentile(zc, 80)
    lo_c = np.nanpercentile(zc, 20)
    hi_k = np.nanpercentile(y, 80)
    lo_k = np.nanpercentile(y, 20)
    out["quadrant"] = np.select(
        [(zc >= hi_c) & (y >= hi_k), (zc >= hi_c) & (y <= lo_k),
         (zc <= lo_c) & (y >= hi_k), (zc <= lo_c) & (y <= lo_k)],
        ["concordant_high", "similar_not_cosensitised",
         "cosensitised_not_similar", "concordant_low"], default="middle")
    print(out.quadrant.value_counts().to_string())

    disc = out[out.quadrant.isin(["similar_not_cosensitised",
                                  "cosensitised_not_similar"])]
    disc.to_csv(OUT_DISCORDANT, index=False)

    cols = ["allergen_a", "allergen_b", "n11", CLINICAL, best,
            "same_family_any", "identity_global_max"]
    for q in ("cosensitised_not_similar", "similar_not_cosensitised"):
        sub = out[out.quadrant == q]
        print(f"\n  {q} ({len(sub)}):")
        print(sub.nlargest(10, CLINICAL if q.startswith("cosens") else best)[cols]
              .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    print(f"\nWritten: {OUT_PAIRS}, {OUT_SUMMARY}, {OUT_DISCORDANT}")
    print("=" * 78)


if __name__ == "__main__":
    main()
