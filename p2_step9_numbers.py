"""
P2 - Stage 9: collect every number the manuscript quotes into one JSON file.

The manuscript builder reads this file and formats the values into the text,
so no result is transcribed by hand into prose. If a stage output changes,
re-running this and the builder updates the manuscript.

Run:
    python p2_step9_numbers.py
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, norm

RAW = "cleaned_allergy_15_sept.xlsx"
MATRIX = "p2_binary_matrix.csv"
CLIN = "p2_pair_table_adjusted.csv"
SETS_RAW = "p2_allergen_sets.csv"
SETS = "p2_allergen_sets_final.csv"
PAIRS = "p2_esm2_pairs.csv"
PROT = "p2_protein_pair_metrics.csv"
DROPPED = "p2_dropped_records.csv"
OUT = "p2_numbers.json"

CLINICAL, PRIMARY, FAMILY = "z_adj", "z_esm2_cos_max", "z_same_family_max"
Q_CUT = 0.10


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        B = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ beta
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def pval(r, n, k=1):
    z = np.arctanh(r) * np.sqrt(n - k - 3)
    return float(2 * (1 - norm.cdf(abs(z))))


def roc_auc(score, label):
    o = np.argsort(-np.asarray(score))
    y = np.asarray(label)[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    tpr = np.concatenate([[0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
    return float(np.trapezoid(tpr, fpr))


def main():
    N = {}

    # ---- cohort -------------------------------------------------------
    B = pd.read_csv(MATRIX, index_col=0)
    atopy = B.sum(axis=1).values
    N["n_patients"] = int(B.shape[0])
    N["n_allergens"] = int(B.shape[1])
    N["n_tests"] = int(B.shape[0] * B.shape[1])
    N["n_sensitised"] = int((atopy >= 1).sum())
    N["pct_sensitised"] = round(100 * (atopy >= 1).mean(), 1)
    N["atopy_mean"] = round(float(atopy.mean()), 2)
    N["atopy_sd"] = round(float(atopy.std(ddof=1)), 2)
    N["atopy_max"] = int(atopy.max())
    prev = B.sum(axis=0).sort_values(ascending=False)
    N["prev_top"] = [{"allergen": k, "n": int(v),
                      "pct": round(100 * v / len(B), 1)}
                     for k, v in prev.head(5).items()]

    tot = float(sum(k * (k - 1) / 2 for k in atopy))
    N["cooc_total"] = int(tot)
    for cut in (5, 10):
        sel = atopy >= cut
        N[f"cooc_share_ge{cut}"] = round(
            100 * sum(k * (k - 1) / 2 for k in atopy[sel]) / tot, 1)
        N[f"n_patients_ge{cut}"] = int(sel.sum())
        N[f"pct_patients_ge{cut}"] = round(100 * sel.mean(), 1)

    # ---- clinical layer ------------------------------------------------
    c = pd.read_csv(CLIN)
    N["n_pairs_all"] = int(len(c))
    N["n_sig_unadj"] = int((c.fdr_q_unadj < 0.05).sum())
    N["n_sig_adj"] = int((c.fdr_q_adj < 0.05).sum())
    N["z_sd"] = round(float(c.z_adj.std()), 3)
    N["rho_phi_z"] = round(float(spearmanr(c.phi, c.z_adj).statistic), 3)
    N["n_z_ge3"] = int((c.z_adj >= 3).sum())
    top = c.nlargest(6, "z_adj")
    N["top_clinical"] = [
        {"a": r.allergen_a.strip(), "b": r.allergen_b.strip(), "n11": int(r.n11),
         "exp": round(float(r.n11_null_mean), 1), "z": round(float(r.z_adj), 2)}
        for r in top.itertuples()]

    # ---- sequence sets -------------------------------------------------
    sr = pd.read_csv(SETS_RAW)
    s = pd.read_csv(SETS)
    dr = pd.read_csv(DROPPED)
    N["n_records_before"] = int(len(sr))
    N["n_records_after"] = int(len(s))
    N["n_proteins"] = int(s.accession.nunique())
    N["n_entries_final"] = int(s.panel_entry.nunique())
    N["n_fragments"] = int(len(dr))
    N["n_pollen_removed"] = int(
        sr.protein_names.str.contains("pollen", case=False, na=False).sum())
    sz_before = sr.groupby("panel_entry").size()
    sz_after = s.groupby("panel_entry").size()
    N["set_min_before"], N["set_max_before"] = int(sz_before.min()), int(sz_before.max())
    N["set_med_before"] = int(sz_before.median())
    N["set_min_after"], N["set_max_after"] = int(sz_after.min()), int(sz_after.max())
    N["set_med_after"] = int(sz_after.median())
    N["n_entries_unnamed"] = int(
        (s.groupby("panel_entry").iuis_own.apply(
            lambda v: v.fillna("").eq("").all())).sum())
    tiers = s.tier.str.split("_").str[0].value_counts().sort_index()
    N["tier_counts"] = {f"tier{k}": int(v) for k, v in tiers.items()}

    # ---- pairs and metrics ---------------------------------------------
    d = pd.read_csv(PAIRS)
    d["allergen_a"] = d.allergen_a.str.strip()
    d["allergen_b"] = d.allergen_b.str.strip()
    y = d[CLINICAL].values
    N["n_pairs_analysis"] = int(len(d))
    N["n_pairs_lost_entry"] = int(N["n_pairs_all"] - len(d) - 3)
    N["n_prot_pairs"] = int(len(pd.read_csv(PROT)))
    N["n_positive"] = int((d.fdr_q_adj < Q_CUT).sum())
    lab = (d.fdr_q_adj < Q_CUT).astype(int).values

    labels = {
        FAMILY: "Shared protein family",
        PRIMARY: "ESM-2 embedding cosine",
        "z_esm2_cos_centered_max": "ESM-2 cosine, centred",
        "z_window80_max": "FAO/WHO 80-aa window",
        "z_kmer6_max": "Identical 6-mers",
        "z_kmer8_max": "Identical 8-mers",
        "z_identity_global_max": "Global sequence identity",
    }
    entries = sorted(set(d.allergen_a) | set(d.allergen_b))
    rows = []
    for col, lbl in labels.items():
        ok = np.isfinite(d[col]) & np.isfinite(d[FAMILY])
        r = spearmanr(d.loc[ok, col], y[ok.values])
        j = []
        for e in entries:
            k = ok & ~((d.allergen_a == e) | (d.allergen_b == e))
            if k.sum() > 30:
                j.append(spearmanr(d.loc[k, col], y[k.values]).statistic)
        rows.append({
            "label": lbl, "col": col, "n": int(ok.sum()),
            "rho": round(float(r.statistic), 3), "p": float(r.pvalue),
            "loao_lo": round(float(min(j)), 3), "loao_hi": round(float(max(j)), 3),
            "partial_family": (None if col == FAMILY else
                               round(partial_spearman(d.loc[ok, col].values,
                                                      y[ok.values],
                                                      d.loc[ok, FAMILY].values), 3)),
            "auc": round(roc_auc(d.loc[ok, col].values, lab[ok.values]), 2),
            "setsize_raw": round(float(spearmanr(
                d[col.replace("z_", "", 1) if col != FAMILY else "same_family_any"],
                d.n_combinations).statistic), 3) if (
                    col.replace("z_", "", 1) in d.columns or col == FAMILY) else None,
            "setsize_adj": round(float(spearmanr(
                d.loc[ok, col], d.loc[ok, "n_combinations"]).statistic), 3),
        })
    rows.sort(key=lambda r: -r["rho"])
    N["metrics"] = rows

    # ---- ESM-2 vs family ------------------------------------------------
    ok = np.isfinite(d[PRIMARY]) & np.isfinite(d[FAMILY])
    n = int(ok.sum())
    e, f = d[PRIMARY].values, d[FAMILY].values
    pe = partial_spearman(e[ok], y[ok.values], f[ok])
    pf = partial_spearman(f[ok], y[ok.values], e[ok])
    N["esm_partial"] = {"rho": round(pe, 3), "p": round(pval(pe, n), 4)}
    N["fam_partial"] = {"rho": round(pf, 3), "p": round(pval(pf, n), 4)}
    je, jf = [], []
    for ent in entries:
        k = ok & ~((d.allergen_a == ent) | (d.allergen_b == ent))
        if k.sum() > 30:
            je.append(partial_spearman(e[k], y[k.values], f[k]))
            jf.append(partial_spearman(f[k], y[k.values], e[k]))
    N["esm_partial"]["loao_lo"] = round(float(min(je)), 3)
    N["esm_partial"]["loao_hi"] = round(float(max(je)), 3)
    N["esm_partial"]["n_sign_flips"] = int(sum(v < 0 for v in je))

    def r2(cols):
        X = np.column_stack([np.ones(n)] + [rankdata(d.loc[ok, c]) for c in cols])
        yy = rankdata(y[ok.values])
        b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        return round(float(1 - ((yy - X @ b) ** 2).sum() /
                           ((yy - yy.mean()) ** 2).sum()), 4)
    N["r2_family"] = r2([FAMILY])
    N["r2_esm"] = r2([PRIMARY])
    N["r2_both"] = r2([PRIMARY, FAMILY])

    rng = np.random.default_rng(20260926)
    xb, yb = d.loc[ok, PRIMARY].values, y[ok.values]
    null = [spearmanr(xb, rng.permutation(yb)).statistic for _ in range(1000)]
    N["negctrl"] = {"mean": round(float(np.mean(null)), 4),
                    "sd": round(float(np.std(null)), 4),
                    "n_exceed": int(np.sum(np.abs(null) >=
                                           abs(spearmanr(xb, yb).statistic)))}

    # ---- discordance ----------------------------------------------------
    x = d[PRIMARY].values
    hx, lx = np.nanpercentile(x, 80), np.nanpercentile(x, 20)
    hy, ly = np.nanpercentile(y, 80), np.nanpercentile(y, 20)
    quad = np.select(
        [(x >= hx) & (y >= hy), (x >= hx) & (y <= ly),
         (x <= lx) & (y >= hy), (x <= lx) & (y <= ly)],
        ["conc_high", "sim_not_cos", "cos_not_sim", "conc_low"], default="middle")
    d = d.assign(q=quad)
    N["quadrants"] = {k: int(v) for k, v in
                      pd.Series(quad).value_counts().items()}

    fam_map = dict(zip(s.accession, s.families.fillna("unassigned")))
    mem = s.groupby("panel_entry").accession.apply(
        lambda v: list(dict.fromkeys(v))).to_dict()
    pp = pd.read_csv(PROT)
    lk = {}
    for r in pp.itertuples():
        lk[(r.acc_a, r.acc_b)] = r.identity_global
        lk[(r.acc_b, r.acc_a)] = r.identity_global
    drv, rows2 = [], []
    for r in d[d.q == "sim_not_cos"].itertuples():
        best = (-1, None, None)
        for a in mem[r.allergen_a]:
            for b in mem[r.allergen_b]:
                if a == b:
                    continue
                v = lk[(a, b)]
                if v > best[0]:
                    best = (v, a, b)
        fam = fam_map[best[1]].split(";")[0].split(",")[0].replace(
            " family", "").strip()
        drv.append(fam)
        rows2.append({"a": r.allergen_a, "b": r.allergen_b,
                      "identity": round(float(best[0]), 3),
                      "z": round(float(getattr(r, CLINICAL)), 2), "family": fam})
    N["driving_family"] = {k: int(v) for k, v in
                           pd.Series(drv).value_counts().items()}
    N["discordant"] = sorted(rows2, key=lambda v: v["z"])[:8]
    ids = [r["identity"] for r in rows2]
    N["sim_identity_min"] = round(min(ids), 2)
    N["sim_identity_max"] = round(max(ids), 2)
    N["cos_not_sim"] = [
        {"a": r.allergen_a, "b": r.allergen_b, "n11": int(r.n11),
         "z": round(float(getattr(r, CLINICAL)), 2)}
        for r in d[d.q == "cos_not_sim"].nlargest(5, CLINICAL).itertuples()]

    # ---- window80 shortcut comparison -----------------------------------
    okw = pp.window80_shortcut.notna()
    N["w80_rho"] = round(float(spearmanr(pp.loc[okw, "window80"],
                                         pp.loc[okw, "window80_shortcut"]).statistic), 3)
    N["w80_maxdiff"] = round(float(
        (pp.loc[okw, "window80_shortcut"] - pp.loc[okw, "window80"]).abs().max()), 3)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(N, fh, indent=1, ensure_ascii=False)
    print(json.dumps(N, indent=1, ensure_ascii=False)[:2600])
    print(f"\nWritten: {OUT}  ({len(N)} top-level keys)")


if __name__ == "__main__":
    main()
