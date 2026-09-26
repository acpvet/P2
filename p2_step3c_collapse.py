"""
P2 - Stage 3c: collapse each panel entry to one protein per allergenic family.

Why this is needed
------------------
Stage 3b filled every panel entry, but at the cost of a defect worse than the
one it fixed. Final set sizes ran from 1 protein (Casein, Onion, Orange,
Sunflower) to 195 (String Bean) - a 195-fold spread that tracks how deeply
each organism happens to be annotated in UniProt, not how many allergens the
food contains. String Bean's 195 are 79 lipid-transfer proteins, 32 vicilins,
28 thaumatins and so on: near-identical paralogues of a handful of families.

This matters because every similarity metric planned for Stage 4 aggregates
over all protein x protein combinations. Max-similarity rises monotonically
with set size, so entries with large sets would score as similar to everything
- and those entries (Pumpkin Seeds, String Bean, Onion, Fig) are precisely the
ones carrying the strongest adjusted clinical signal. The bias would run
straight through the result.

The fix
-------
Cross-reactivity operates between members of the same protein family: LTP with
LTP, profilin with profilin, 2S albumin with 2S albumin. Keeping 79 copies of
one family adds no information that one copy does not carry. Each panel entry
is therefore collapsed to one representative per distinct allergen, where
"distinct allergen" means a WHO/IUIS designation where one exists and a UniProt
protein family otherwise.

Representative within a group, in order:
  1. carries a WHO/IUIS designation;
  2. among those, the sequence whose length is closest to the group median
     (a fragment or a fusion artefact is never the median).

Two further rules:
  - The panel tests olive as a FRUIT (confirmed). Proteins named as pollen
    allergens are removed from every entry, not only olive - rice Ory s 1 is
    an expansin from pollen and has the same problem.
  - A WHO/IUIS designation whose genus code cannot be generated from the
    organism's own name is an annotation transferred by similarity from
    another species, not a designation for this organism. Cucurbita maxima
    carrying "Pru ar 1" is the example. Such names are treated as absent.

Run:
    python p2_step3c_collapse.py

Requires: pandas, numpy.  No internet.  Runtime seconds.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
SETS_PATH = "p2_allergen_sets.csv"
FASTA_PATH = "p2_sequences_final.fasta"

DROP_POLLEN = True                 # panel is a food panel
POLLEN_MARKERS = ["pollen"]

OUT_SETS = "p2_allergen_sets_final.csv"
OUT_FASTA = "p2_sequences_curated.fasta"
OUT_REPORT = "p2_set_composition.csv"
# --------------------------------------------------------------------------


def read_fasta(path):
    seqs, acc, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
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


def genus_codes(scientific_name):
    """Genus part of the WHO/IUIS designation, generated from the organism
    name. Only the genus is testable: the species letter often comes from an
    older binomial than the one UniProt now uses - Bos d from Bos domesticus
    while UniProt says Bos taurus, Gal d from Gallus domesticus, Mus xp from
    Musa x paradisiaca. Requiring the species letter to match would reject
    Bos d 8 and Gal d 2, which are correct."""
    parts = str(scientific_name).split()
    if not parts:
        return set()
    g = parts[0]
    return {g[:3].capitalize(), g[:4].capitalize()}


def merge_synonyms(names_per_record):
    """Designations appearing together on one record are the same allergen
    (Pru a 1 and Pru av 1). Union them so they collapse into one group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for names in names_per_record:
        if len(names) > 1:
            root = find(names[0])
            for n in names[1:]:
                parent[find(n)] = root
    return {n: find(n) for n in parent}


def main():
    print("=" * 78)
    print("P2 - STAGE 3c: ONE REPRESENTATIVE PER ALLERGENIC FAMILY")
    print("=" * 78)

    d = pd.read_csv(SETS_PATH)
    seqs = read_fasta(FASTA_PATH)
    missing = set(d.accession) - set(seqs)
    assert not missing, f"{len(missing)} accessions have no sequence: {list(missing)[:5]}"
    d["seq"] = d.accession.map(seqs)
    d["seq_len"] = d.seq.str.len()
    bad = d[d.seq_len != d.length]
    assert bad.empty, (f"Length column disagrees with the FASTA for "
                       f"{len(bad)} records, e.g. {bad.accession.tolist()[:5]}")
    print(f"\nInput: {len(d)} rows, {d.accession.nunique()} unique proteins, "
          f"{d.panel_entry.nunique()} panel entries")
    print(f"Set sizes before: min {d.groupby('panel_entry').size().min()}  "
          f"median {int(d.groupby('panel_entry').size().median())}  "
          f"max {d.groupby('panel_entry').size().max()}")

    # ---- rule 1: drop pollen allergens from a food panel -----------------
    if DROP_POLLEN:
        is_pollen = d.protein_names.str.contains("|".join(POLLEN_MARKERS),
                                                 case=False, na=False)
        per_entry = d[is_pollen].groupby("panel_entry").size().sort_values(ascending=False)
        print(f"\nPollen-named proteins removed: {int(is_pollen.sum())}")
        for e, n in per_entry.items():
            print(f"    {e:18s} -{n}")
        d = d[~is_pollen].copy()

    # ---- rule 2: reject designations foreign to the organism -------------
    # Re-parse designations from the protein-name text with both orderings.
    # Stage 3b matched only "... allergen Pis v 3"; UniProt also writes
    # "Pis v 1 allergen 2S albumin", which is how Pis v 1, Pis v 2 and Pis v 5
    # are named. Reading the stored column would keep those three unnamed.
    post = re.compile(r"[Aa]llergen\s+([A-Z][a-z]{1,3}\s+[a-z]{1,2}\s+\d+)")
    pre = re.compile(r"\b([A-Z][a-z]{1,3}\s+[a-z]{1,2}\s+\d+)(?:\.\d+)?\s+allergen")
    d["iuis_name"] = [
        "; ".join(sorted({re.sub(r"\s+", " ", m)
                          for m in post.findall(t) + pre.findall(t)}))
        for t in d.protein_names.fillna("")]

    foreign = []

    def own_name(row):
        name = str(row.iuis_name)
        if not name or name == "nan":
            return ""
        # Tiers 1 and 2 are UniProt's own allergen annotation on an entry of
        # this organism; that is authoritative even when the genus code comes
        # from a superseded binomial (Lyc e 1 for Solanum lycopersicum, ex
        # Lycopersicon; Lit v 2 for Penaeus vannamei, ex Litopenaeus; Bos d
        # for Bos domesticus). Only names picked up by the name sweep or the
        # family sweep are tested, because those are where a designation
        # transferred from another species can enter.
        if str(row.tier).startswith(("1", "2")):
            return name
        ok = genus_codes(row.source_organism)
        keep = [n for n in name.split("; ") if n.split()[0] in ok]
        if not keep:
            foreign.append((row.panel_entry, row.accession, name,
                            row.source_organism, row.tier))
        return "; ".join(keep)

    d["iuis_own"] = [own_name(r) for r in d.itertuples()]
    if foreign:
        print(f"\nDesignations rejected as transferred from another species "
              f"(tier 3/4 only): {len(foreign)}")
        for e, a, n, o, tr in foreign[:12]:
            print(f"    {e:16s} {a:12s} {n:12s} {o:22s} tier {tr}")

    # ---- collapse --------------------------------------------------------
    syn = merge_synonyms([n.split("; ") for n in d.iuis_own if n])
    d["group"] = [
        (syn.get(r.iuis_own.split("; ")[0], r.iuis_own.split("; ")[0])
         if r.iuis_own else (r.families if isinstance(r.families, str)
                             and r.families else "UNASSIGNED"))
        for r in d.itertuples()]
    fam_median = d.groupby("families").length.median()

    keep_idx, report = [], []
    for (entry, grp), sub in d.groupby(["panel_entry", "group"], sort=False):
        named = sub[sub.iuis_own != ""]
        pool = named if len(named) else sub
        med = fam_median.get(pool.families.iloc[0], pool.length.median())
        pick = pool.iloc[(pool.length - med).abs().argsort().iloc[0]]
        keep_idx.append(pick.name)
        report.append({"panel_entry": entry, "group": grp,
                       "n_collapsed": len(sub), "kept": pick.accession,
                       "kept_name": pick.iuis_own or "(unnamed)",
                       "kept_len": int(pick.length),
                       "family": pick.families})

    out = d.loc[keep_idx].drop(columns=["seq", "seq_len", "group"])
    rep = pd.DataFrame(report)

    # entries whose whole set was unnamed tier-4 material keep only families
    # that actually exist elsewhere as named allergens - reported, not dropped
    named_families = set(d.loc[d.iuis_own != "", "families"].dropna())
    out["family_is_allergenic_elsewhere"] = out.families.isin(named_families)

    out.to_csv(OUT_SETS, index=False)
    rep.to_csv(OUT_REPORT, index=False)
    with open(OUT_FASTA, "w") as fh:
        for acc in sorted(set(out.accession)):
            fh.write(f">{acc}\n")
            s = seqs[acc]
            for i in range(0, len(s), 60):
                fh.write(s[i:i+60] + "\n")

    sizes = out.groupby("panel_entry").size()
    print("\n=== SET SIZE AFTER COLLAPSE ===")
    print(f"  min {sizes.min()}   median {int(sizes.median())}   "
          f"max {sizes.max()}   total rows {len(out)}   "
          f"unique proteins {out.accession.nunique()}")

    print("\n=== FINAL COMPOSITION PER PANEL ENTRY ===")
    for entry in sorted(out.panel_entry.unique()):
        sub = out[out.panel_entry == entry]
        names = [n for n in sub.iuis_own if n]
        fams = [f for f, n in zip(sub.families.fillna("?"), sub.iuis_own)
                if not n]
        shown = ", ".join(names)
        if fams:
            shown += (" | unnamed: " +
                      ", ".join(f.replace(" family", "") for f in fams[:5]))
        n_before = int((d.panel_entry == entry).sum())
        print(f"  {entry:18s} {len(sub):2d} (from {n_before:3d})  {shown[:88]}")

    print("\n=== BIGGEST COLLAPSES ===")
    print(rep.nlargest(10, "n_collapsed")[
        ["panel_entry", "group", "n_collapsed", "kept", "kept_len"]]
        .to_string(index=False))

    print("\n=== SET-SIZE CHECK ===")
    print("  Stage 4 must confirm that the chosen similarity metric does not")
    print("  correlate with set size. Sizes now:")
    print("   " + "  ".join(f"{e}:{n}" for e, n in sizes.sort_values().items()))

    print(f"\nWritten: {OUT_SETS}, {OUT_FASTA}, {OUT_REPORT}")
    print("=" * 78)


if __name__ == "__main__":
    main()
