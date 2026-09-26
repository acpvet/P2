"""
P2 - Stage 2: the adjusted pairwise clinical layer.

Input : cleaned_allergy_15_sept.xlsx   (raw long format, 13770 rows)
        jordan_sensitization.xlsx      (aggregated workbook - used only to
                                        cross-validate, set to None to skip)

The question this stage answers
-------------------------------
13 patients (3.2% of the cohort) generate 65% of all co-sensitization events.
A raw phi or odds ratio between two allergens is therefore partly a restatement
of "these people react to everything". This stage removes that and asks which
pairs co-occur MORE than expected once BOTH the patient's total degree of
sensitization AND the allergen's prevalence are held fixed.

Method: curveball randomisation (Strona et al., Nat Commun 2014). Every
permuted matrix has exactly the same row sums (per-patient atopy index) and
exactly the same column sums (per-allergen prevalence) as the observed data.
The null distribution of n11 is built per pair from N_PERM such matrices.
Nothing parametric is assumed. Rows that are all-zero or all-one cannot take
part in a trade and are excluded from the chain; they contribute zero to every
co-occurrence count, observed and permuted alike.

The adjusted quantity used downstream is z_adj = (n11 - E[n11]) / SD[n11].
It is continuous, so the linkage step in Stage 4 does not depend on any
per-pair significance call.

Run:
    python p2_step2_adjusted.py

Requires: pandas, numpy, scipy, openpyxl.   Runtime ~3 min.
"""

import math
import random
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# --------------------------------------------------------------------------
RAW_PATH = "cleaned_allergy_15_sept.xlsx"
AGG_PATH = "jordan_sensitization.xlsx"    # None to skip cross-validation

CLASS_POSITIVE = 1         # CAP class >= 1 counts as sensitized
N_PERM = 50000
BURN_IN_FACTOR = 5         # trades per permutation = factor * n_active_rows
SEED = 20260925

OUT_MATRIX = "p2_binary_matrix.csv"
OUT_PAIRS = "p2_pair_table_adjusted.csv"
# --------------------------------------------------------------------------


def load_raw(path):
    d = pd.read_excel(path, usecols="A:H")
    d.columns = [str(c).strip() for c in d.columns]
    needed = ["Allergen name", "Class", "Patient Number",
              "Gender", "Age Group", "s-IgE Status"]
    missing = [c for c in needed if c not in d.columns]
    assert not missing, f"Missing columns in raw file: {missing}"

    d["Allergen name"] = d["Allergen name"].astype(str).str.strip()
    # Patient IDs are mixed int/str in the source file (one lab accession,
    # '124182I', sits among integers). Normalise to str so nothing is
    # silently coerced or two IDs collapsed into one.
    d["Patient Number"] = d["Patient Number"].astype(str).str.strip()

    pos = d["s-IgE Status"].astype(str).str.strip().str.lower().eq("positive")
    assert (pos == (d["Class"] >= CLASS_POSITIVE)).all(), \
        "s-IgE Status disagrees with Class on at least one row"

    g = d.groupby("Patient Number").agg(sex=("Gender", "nunique"),
                                        grp=("Age Group", "nunique"))
    assert (g.sex == 1).all(), "A patient has more than one Gender"
    assert (g.grp == 1).all(), "A patient has more than one Age Group"
    assert not d.duplicated(["Patient Number", "Allergen name"]).any(), \
        "Duplicate patient x allergen rows"
    return d


def build_matrix(d):
    wide = d.pivot(index="Patient Number", columns="Allergen name", values="Class")
    assert wide.notna().all().all(), "Incomplete grid - a patient x allergen is missing"
    return (wide >= CLASS_POSITIVE).astype(np.int8)


def validate_against_aggregate(B, agg_path):
    """The aggregated workbook is an independent computation over the same
    data. Disagreement means one of the two files is wrong."""
    summ = pd.read_excel(agg_path, "Allergen_Summary")
    summ["Allergen"] = summ["Allergen"].astype(str).str.strip()
    ref = dict(zip(summ["Allergen"], summ["N_positive"].astype(int)))
    bad = {k: (int(v), ref.get(k)) for k, v in B.sum(axis=0).items()
           if ref.get(k) != int(v)}
    assert not bad, f"Prevalence disagrees with aggregated workbook: {bad}"

    corr = pd.read_excel(agg_path, "CoSensitization_Corr", index_col=0)
    corr.index = [str(i).strip() for i in corr.index]
    corr.columns = [str(c).strip() for c in corr.columns]
    R = np.corrcoef(B.values.astype(float), rowvar=False)
    cols = list(B.columns)
    worst = max(abs(R[i, j] - float(corr.loc[cols[i], cols[j]]))
                for i, j in combinations(range(len(cols)), 2))
    assert worst < 1e-8, f"phi disagrees with aggregated workbook (max {worst:.2e})"
    print(f"Cross-validation vs aggregated workbook: {len(cols)}/{len(cols)} "
          f"prevalences and {len(cols)*(len(cols)-1)//2} phi values agree "
          f"(max residual {worst:.2e})")


def set_bits(x):
    out = []
    while x:
        low = x & -x
        out.append(low)
        x ^= low
    return out


def curveball_pass(masks, rnd, n_trades):
    """One pass of n_trades checkerboard swaps. Row and column sums invariant."""
    n = len(masks)
    for _ in range(n_trades):
        i = rnd.randrange(n)
        j = rnd.randrange(n)
        if i == j:
            continue
        a, b = masks[i], masks[j]
        only_a = a & ~b
        if only_a == 0:
            continue
        only_b = b & ~a
        if only_b == 0:
            continue
        common = a & b
        pool = set_bits(only_a | only_b)
        rnd.shuffle(pool)
        k = bin(only_a).count("1")
        na = nb = common
        for v in pool[:k]:
            na |= v
        for v in pool[k:]:
            nb |= v
        masks[i], masks[j] = na, nb
    return masks


def densify(masks, shifts):
    arr = np.array(masks, dtype=np.uint64)
    return ((arr[:, None] >> shifts) & 1).astype(np.int32)


def bh_fdr(p):
    p = np.asarray(p, float)
    m = len(p)
    q = np.empty(m)
    run = 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1]):
        run = min(run, p[idx] * m / (m - rank))
        q[idx] = run
    return q


def main():
    print("=" * 78)
    print("P2 - STAGE 2: MARGIN-PRESERVING ADJUSTMENT OF THE CLINICAL LAYER")
    print("=" * 78)

    d = load_raw(RAW_PATH)
    B = build_matrix(d)
    names = list(B.columns)
    X = B.values.astype(np.int32)
    n_pat, n_all = X.shape
    n_pairs = n_all * (n_all - 1) // 2
    print(f"\nPatients {n_pat}   Allergens {n_all}   Pairs {n_pairs}")
    if AGG_PATH:
        validate_against_aggregate(B, AGG_PATH)
    B.to_csv(OUT_MATRIX)

    atopy = X.sum(axis=1)
    print(f"\nAtopy index: mean {atopy.mean():.2f}  SD {atopy.std(ddof=1):.2f}  "
          f"median {int(np.median(atopy))}  max {atopy.max()}")
    print(f"  sensitized (>=1) {int((atopy >= 1).sum())}   "
          f"non-sensitized {int((atopy == 0).sum())}")

    obs = X.T @ X
    col = X.sum(axis=0)

    active = [r for r in range(n_pat) if 0 < atopy[r] < n_all]
    masks = [int(sum(int(X[r, c]) << c for c in range(n_all))) for r in active]
    shifts = np.arange(n_all, dtype=np.uint64)
    assert (densify(masks, shifts).sum(axis=0) == col).all(), \
        "Bitmask encoding lost a positive"

    n_trades = BURN_IN_FACTOR * len(masks)
    rnd = random.Random(SEED)
    print(f"\nCurveball randomisation: {N_PERM} permutations x {n_trades} trades, "
          f"seed {SEED}")
    print(f"  rows in the chain: {len(masks)} (all-zero rows cannot trade and "
          f"contribute nothing)")
    print("  row sums (atopy index) and column sums (prevalence) held exact")

    masks = curveball_pass(masks, rnd, n_trades * 20)     # burn-in

    ge = np.zeros((n_all, n_all), dtype=np.int64)
    s1 = np.zeros((n_all, n_all))
    s2 = np.zeros((n_all, n_all))
    step = max(1, N_PERM // 10)
    for it in range(N_PERM):
        masks = curveball_pass(masks, rnd, n_trades)
        Xp = densify(masks, shifts)
        C = Xp.T @ Xp
        ge += (C >= obs)
        s1 += C
        s2 += C.astype(float) ** 2
        if (it + 1) % step == 0:
            print(f"  ... {it+1}/{N_PERM}")
    assert (densify(masks, shifts).sum(axis=0) == col).all(), "Column margin broken"

    mean = s1 / N_PERM
    sd = np.sqrt(np.maximum(s2 / N_PERM - mean ** 2, 0.0))

    rows = []
    for i, j in combinations(range(n_all), 2):
        a, b = int(col[i]), int(col[j])
        n11 = int(obs[i, j])
        n10, n01 = a - n11, b - n11
        n00 = n_pat - a - b + n11
        phi = ((n11 * n00 - n10 * n01) /
               math.sqrt(a * (n_pat - a) * b * (n_pat - b)))
        odds = ((n11 + .5) * (n00 + .5)) / ((n10 + .5) * (n01 + .5))
        rows.append({
            "allergen_a": names[i], "allergen_b": names[j],
            "n_pos_a": a, "n_pos_b": b,
            "n11": n11, "n10": n10, "n01": n01, "n00": n00,
            "phi": phi, "odds_ratio": odds,
            "fisher_p": fisher_exact([[n11, n10], [n01, n00]],
                                     alternative="greater")[1],
            "n11_null_mean": mean[i, j], "n11_null_sd": sd[i, j],
            "excess": n11 - mean[i, j],
            "z_adj": (n11 - mean[i, j]) / sd[i, j] if sd[i, j] > 0 else np.nan,
            "p_perm": (ge[i, j] + 1) / (N_PERM + 1)})

    t = pd.DataFrame(rows)
    t["fdr_q_unadj"] = bh_fdr(t["fisher_p"].values)
    t["fdr_q_adj"] = bh_fdr(t["p_perm"].values)
    t = t.sort_values("z_adj", ascending=False)
    t.to_csv(OUT_PAIRS, index=False)

    floor = 1.0 / (N_PERM + 1)
    n_floor = int((t.p_perm <= floor + 1e-12).sum())
    print("\n=== RESOLUTION ===")
    print(f"  smallest attainable permutation p = {floor:.2e}; "
          f"{n_floor} pairs sit at that floor")

    n_un = int((t.fdr_q_unadj < 0.05).sum())
    n_ad = int((t.fdr_q_adj < 0.05).sum())
    kept = int(((t.fdr_q_unadj < 0.05) & (t.fdr_q_adj < 0.05)).sum())
    gained = int(((t.fdr_q_unadj >= 0.05) & (t.fdr_q_adj < 0.05)).sum())
    print("\n=== EFFECT OF THE ADJUSTMENT ===")
    print(f"  unadjusted significant (Fisher + BH, q<0.05) : {n_un} / {n_pairs}")
    print(f"  adjusted   significant (curveball + BH, q<0.05): {n_ad} / {n_pairs}")
    print(f"    survived {kept}   lost {n_un - kept}   revealed only after {gained}")
    print(f"  Spearman(phi_unadjusted, z_adjusted) = "
          f"{t[['phi','z_adj']].corr(method='spearman').iloc[0,1]:.3f}")
    print(f"  z_adj: mean {t.z_adj.mean():.3f}  SD {t.z_adj.std():.3f}  "
          f"(null expectation 0 and 1)")
    for c in (2, 3, 4):
        print(f"    pairs with z_adj >= {c}: {int((t.z_adj >= c).sum())}   "
              f"<= -{c}: {int((t.z_adj <= -c).sum())}")

    print("\n=== TOP 20 AFTER ADJUSTMENT ===")
    print(t.head(20)[["allergen_a", "allergen_b", "n11", "n11_null_mean",
                      "excess", "z_adj", "phi", "fdr_q_adj"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== LARGEST DEMOTIONS (unadjusted-significant, lowest z_adj) ===")
    print(t[t.fdr_q_unadj < 0.05].nsmallest(10, "z_adj")[
        ["allergen_a", "allergen_b", "n11", "n11_null_mean", "z_adj",
         "phi", "fdr_q_adj"]]
        .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== GO / NO-GO FOR THE LINKAGE STEP ===")
    strong = int((t.z_adj >= 3).sum())
    if strong < 15:
        print(f"  STOP: only {strong} pairs exceed z_adj = 3. The clinical "
              f"signal beyond overall atopy is too thin to correlate against "
              f"computational metrics. Redesign before any sequence work.")
    else:
        mdr = math.tanh(2.801585 / math.sqrt(n_pairs - 3))
        print(f"  PROCEED: {strong} pairs exceed z_adj = 3, "
              f"{n_ad} pass FDR < 0.05.")
        print(f"  Stage 4 uses z_adj over all {n_pairs} pairs as the outcome; "
              f"minimum detectable rho = {mdr:.3f}.")

    print(f"\nWritten: {OUT_MATRIX}, {OUT_PAIRS}")
    print("=" * 78)


if __name__ == "__main__":
    main()
