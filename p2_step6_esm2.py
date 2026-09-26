"""
P2 - Stage 6: does a protein language model add anything over a family label?

Stage 5 found that every sequence metric - global identity, the FAO/WHO 80-aa
window, exact 6/8-mer sharing - drops to zero once family membership is held
fixed. They are less efficient restatements of "these two proteins belong to
the same family".

That leaves one question worth asking. A pLM embedding is not a sequence
identity: it encodes structural and functional context that a family label
does not carry. If ESM-2 also adds nothing over the family label, the
conclusion stops being about weak metrics and becomes a statement about the
ceiling of sequence-based cross-reactivity prediction against real clinical
co-sensitization.

Everything downstream of the embedding is identical to Stage 5 - same
relabelling null, same z-scores, same leave-one-allergen-out, same negative
control - so the new metric is directly comparable to the old ones.

Two cosine variants are computed, because mean-pooled pLM embeddings are
anisotropic: unrelated proteins sit at cosine ~0.9 simply because all
embeddings occupy a narrow cone. The centred variant subtracts the mean
embedding first, which removes that common direction. The output reports both
and says which one was used.

Sequences longer than the model's context are chunked with overlap and
length-weighted averaged; the run prints every sequence this applied to
rather than truncating silently.

Run:
    python p2_step6_esm2.py

Requires: pandas, numpy, scipy, torch, transformers.
    pip install transformers
    pip install torch --index-url https://download.pytorch.org/whl/cpu
CPU is fine: 155 sequences. First run downloads the model (~2.5 GB for 650M).
Embeddings are cached, so a re-run skips straight to the analysis.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

# --------------------------------------------------------------------------
SETS_PATH = "p2_allergen_sets_final.csv"
FASTA_PATH = "p2_sequences_curated.fasta"
LINKAGE_PATH = "p2_linkage_pairs.csv"       # from Stage 5, carries z_adj

MODEL = "facebook/esm2_t33_650M_UR50D"      # or esm2_t30_150M_UR50D if slow
MAX_TOKENS = 1022                           # model context minus BOS/EOS
CHUNK_OVERLAP = 128
CACHE = "p2_esm2_embeddings.npz"

METRICS = ["esm2_cos", "esm2_cos_centered"]
TOP_K = 3
N_PERM = 10000
SEED = 20260925
CLINICAL = "z_adj"
FAMILY_Z = "z_same_family_max"               # the metric to beat

OUT_PAIRS = "p2_esm2_pairs.csv"
OUT_SUMMARY = "p2_esm2_summary.csv"
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


def embed_all(prot, seqs):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["accessions"]) == list(prot) and str(z["model"]) == MODEL:
            print(f"Embeddings loaded from {CACHE} ({z['E'].shape})")
            return z["E"]
        print(f"{CACHE} does not match this protein set or model - recomputing.")

    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading {MODEL} (first run downloads it)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    torch.set_grad_enabled(False)

    chunked = []
    vecs = []
    for i, acc in enumerate(prot, 1):
        s = seqs[acc]
        pieces, start = [], 0
        while start < len(s):
            end = min(start + MAX_TOKENS, len(s))
            pieces.append(s[start:end])
            if end == len(s):
                break
            start = end - CHUNK_OVERLAP
        if len(pieces) > 1:
            chunked.append((acc, len(s), len(pieces)))
        reps, weights = [], []
        for p in pieces:
            enc = tok(p, return_tensors="pt", add_special_tokens=True)
            out = model(**enc).last_hidden_state[0]
            # drop BOS and EOS, mean over residues
            reps.append(out[1:-1].mean(dim=0).numpy())
            weights.append(len(p))
        w = np.array(weights, dtype=float)
        vecs.append(np.average(np.stack(reps), axis=0, weights=w))
        if i % 25 == 0 or i == len(prot):
            print(f"  embedded {i}/{len(prot)}")

    E = np.stack(vecs)
    np.savez(CACHE, E=E, accessions=np.array(prot), model=MODEL)
    if chunked:
        print(f"\nSequences longer than {MAX_TOKENS} residues, chunked and "
              f"length-weighted averaged:")
        for acc, L, n in chunked:
            print(f"    {acc}  {L} aa  -> {n} chunks")
    else:
        print(f"\nNo sequence exceeded {MAX_TOKENS} residues.")
    return E


def cosine_matrix(E):
    n = E / np.linalg.norm(E, axis=1, keepdims=True)
    return n @ n.T


def aggregate(M, ia, ib):
    sub = M[:, ia][:, :, ib].reshape(M.shape[0], -1)
    mx = sub.max(axis=1)
    k = min(TOP_K, sub.shape[1])
    return mx, (-np.partition(-sub, k - 1, axis=1)[:, :k]).mean(axis=1)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b, a, rcond=None)
        return a - b @ beta
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def main():
    print("=" * 78)
    print("P2 - STAGE 6: ESM-2 EMBEDDING SIMILARITY")
    print("=" * 78)

    sets = pd.read_csv(SETS_PATH)
    seqs = read_fasta(FASTA_PATH)
    pairs = pd.read_csv(LINKAGE_PATH)
    assert CLINICAL in pairs.columns and FAMILY_Z in pairs.columns, \
        f"{LINKAGE_PATH} must carry {CLINICAL} and {FAMILY_Z} from Stage 5"

    entries = sorted(sets.panel_entry.unique())
    prot = sorted(set(sets.accession))
    idx = {a: i for i, a in enumerate(prot)}
    members = {e: np.array([idx[a] for a in
                            sets.loc[sets.panel_entry == e, "accession"].unique()])
               for e in entries}
    print(f"\nProteins {len(prot)}   panel entries {len(entries)}   "
          f"pairs {len(pairs)}")

    E = embed_all(prot, seqs)
    assert E.shape[0] == len(prot), "embedding count does not match proteins"

    C_raw = cosine_matrix(E)
    C_cen = cosine_matrix(E - E.mean(axis=0, keepdims=True))
    off = ~np.eye(len(prot), dtype=bool)
    print(f"\n=== COSINE DISTRIBUTIONS (off-diagonal) ===")
    for tag, C in (("raw", C_raw), ("centred", C_cen)):
        v = C[off]
        print(f"  {tag:8s} min {v.min():+.3f}  q25 {np.percentile(v,25):+.3f}  "
              f"median {np.median(v):+.3f}  q75 {np.percentile(v,75):+.3f}  "
              f"max {v.max():+.3f}")
    spread = np.percentile(C_raw[off], 75) - np.percentile(C_raw[off], 25)
    print(f"  raw interquartile spread {spread:.3f} - a narrow spread is the "
          f"anisotropy the centred variant removes.")

    M = np.stack([C_raw, C_cen])
    ia_list = [members[r.allergen_a.strip()] for r in pairs.itertuples()]
    ib_list = [members[r.allergen_b.strip()] for r in pairs.itertuples()]

    obs_mx, obs_tk = [], []
    for ia, ib in zip(ia_list, ib_list):
        mx, tk = aggregate(M, ia, ib)
        obs_mx.append(mx); obs_tk.append(tk)
    obs_mx, obs_tk = np.array(obs_mx), np.array(obs_tk)

    rng = np.random.default_rng(SEED)
    s1m = np.zeros_like(obs_mx); s2m = np.zeros_like(obs_mx)
    s1t = np.zeros_like(obs_tk); s2t = np.zeros_like(obs_tk)
    print(f"\nRelabelling null: {N_PERM} permutations, seed {SEED} "
          f"(identical to Stage 5)")
    step = max(1, N_PERM // 10)
    for it in range(N_PERM):
        p = rng.permutation(len(prot))
        for i, (ia, ib) in enumerate(zip(ia_list, ib_list)):
            mx, tk = aggregate(M, p[ia], p[ib])
            s1m[i] += mx; s2m[i] += mx ** 2
            s1t[i] += tk; s2t[i] += tk ** 2
        if (it + 1) % step == 0:
            print(f"  ... {it+1}/{N_PERM}")

    out = pairs.copy()
    zcols = []
    for m, name in enumerate(METRICS):
        for tag, obs, s1, s2 in (("max", obs_mx[:, m], s1m[:, m], s2m[:, m]),
                                 (f"top{TOP_K}", obs_tk[:, m], s1t[:, m],
                                  s2t[:, m])):
            mu = s1 / N_PERM
            sd = np.sqrt(np.maximum(s2 / N_PERM - mu ** 2, 0.0))
            c = f"z_{name}_{tag}"
            out[c] = np.where(sd > 0, (obs - mu) / np.where(sd > 0, sd, 1.0),
                              np.nan)
            out[f"raw_{name}_{tag}"] = obs
            zcols.append(c)
    out.to_csv(OUT_PAIRS, index=False)

    y = out[CLINICAL].values
    fam = out[FAMILY_Z].values

    print("\n=== SET-SIZE CHECK ===")
    for c in zcols:
        ok = np.isfinite(out[c])
        r = spearmanr(out.loc[ok, c], out.loc[ok, "n_combinations"]).statistic
        print(f"    {c:28s} vs combinations {r:+.3f}"
              f"{'   <-- biased' if abs(r) > 0.3 else ''}")

    print(f"\n=== LINKAGE vs {CLINICAL} ===")
    rows = []
    for c in zcols:
        ok = np.isfinite(out[c]) & np.isfinite(fam)
        r = spearmanr(out.loc[ok, c], y[ok])
        rows.append({
            "metric": c, "n": int(ok.sum()), "rho": r.statistic, "p": r.pvalue,
            "rho_partial_setsize": partial_spearman(
                out.loc[ok, c].values, y[ok], out.loc[ok, "n_combinations"].values),
            "rho_partial_family": partial_spearman(
                out.loc[ok, c].values, y[ok], fam[ok])})
    summ = pd.DataFrame(rows).sort_values("rho", ascending=False)

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
    summ.to_csv(OUT_SUMMARY, index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    best = summ.metric.iloc[0]
    okb = np.isfinite(out[best])
    rng2 = np.random.default_rng(SEED + 1)
    xb, yb = out.loc[okb, best].values, y[okb.values]
    nullr = [spearmanr(xb, rng2.permutation(yb)).statistic for _ in range(1000)]
    print(f"\n=== NEGATIVE CONTROL ===")
    print(f"  {best}: observed {summ.rho.iloc[0]:+.3f}   shuffled mean "
          f"{np.mean(nullr):+.4f} SD {np.std(nullr):.4f}   "
          f"|rho| >= observed in {int(np.sum(np.abs(nullr) >= abs(summ.rho.iloc[0])))}/1000")

    print("\n=== THE QUESTION THIS STAGE EXISTS TO ANSWER ===")
    ok = np.isfinite(out[best]) & np.isfinite(fam)
    r_alone = spearmanr(out.loc[ok, best], y[ok]).statistic
    r_partial = partial_spearman(out.loc[ok, best].values, y[ok], fam[ok])
    r_fam = spearmanr(fam[ok], y[ok]).statistic
    r_fam_partial = partial_spearman(fam[ok], y[ok], out.loc[ok, best].values)
    print(f"  family label alone                      rho {r_fam:+.3f}")
    print(f"  ESM-2 ({best}) alone                    rho {r_alone:+.3f}")
    print(f"  ESM-2 with family partialled out        rho {r_partial:+.3f}")
    print(f"  family with ESM-2 partialled out        rho {r_fam_partial:+.3f}")
    if abs(r_partial) < 0.08:
        print("\n  ESM-2 adds nothing beyond the family label. The ceiling is "
              "not the metric - it is what sequence can say about clinical "
              "co-sensitization.")
    elif r_partial > 0:
        print("\n  ESM-2 carries signal the family label does not. Report both "
              "and keep the pLM layer in the model.")
    else:
        print("\n  ESM-2 runs against the clinical signal once family is fixed "
              "- inspect before interpreting.")

    print("\n=== ALL METRICS, SIDE BY SIDE ===")
    prev = [c for c in out.columns
            if c.startswith("z_") and c not in zcols and c != CLINICAL]
    allrows = []
    for c in prev + zcols:
        ok = np.isfinite(out[c]) & np.isfinite(fam)
        allrows.append({"metric": c,
                        "rho": spearmanr(out.loc[ok, c], y[ok]).statistic,
                        "partial_family": partial_spearman(
                            out.loc[ok, c].values, y[ok], fam[ok])})
    print(pd.DataFrame(allrows).sort_values("rho", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print(f"\nWritten: {OUT_PAIRS}, {OUT_SUMMARY}, {CACHE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
