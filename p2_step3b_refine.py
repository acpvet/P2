"""
P2 - Stage 3b: finalise the allergen sequence set behind each panel entry.

Stage 3 exposed four defects. This stage fixes each one and reports what the
fix recovered.

  1. WHO/IUIS parsing missed 4-letter genus codes ("Sola t 4", "Pru av 1"),
     because the pattern required exactly three. Widened to 2-4.
  2. reviewed + allergen-keyword is too strict. Pistachio returned only
     Pis v 3; Pis v 1 (the 2S albumin) exists in UniProt as unreviewed with no
     allergen keyword. Retrieval is now tiered, and each protein carries the
     tier it came from.
  3. Retrieved records include peptide fragments as short as 11 aa (Gad m 2,
     Thu a 3, Lit v 3). Alignment against a fragment is meaningless, so a
     minimum length applies and every exclusion is listed.
  4. Four panel entries resolved to the same organism as another and would
     therefore score similarity 1.0 by construction. They are split into
     sub-extracts by rules over the retrieved protein names - never by typed
     accessions - and any rule that matches nothing stops the script.

Tier 3 needs no typed allergen names: the WHO/IUIS code is generated from the
organism name by the nomenclature rule itself (first 3-4 letters of the genus
plus the first 1-2 of the species), so Pistacia vera yields "Pis v"/"Pist v",
Solanum tuberosum yields "Sol t"/"Sola t". The taxonomy filter keeps the
resulting query specific.

Run:
    python p2_step3b_refine.py

Requires: pandas, requests.  Needs internet. Runtime ~3 min.
"""

import re
import sys
import time
from collections import defaultdict

import pandas as pd
import requests

# --------------------------------------------------------------------------
SOURCES_PATH = "p2_panel_sources.csv"     # written by Stage 3; taxids read, not retyped
MIN_LENGTH = 50                           # shorter records are fragments
INCLUDE_TIER3 = True                      # name-based sweep
INCLUDE_TIER4 = True                      # pan-allergen family sweep
PAUSE = 0.3

OUT_SETS = "p2_allergen_sets.csv"
OUT_FASTA = "p2_sequences_final.fasta"
OUT_DROPPED = "p2_dropped_records.csv"

KB_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,protein_name,organism_name,length,protein_families,sequence"
EXPECTED_HEADER = ["Entry", "Protein names", "Organism", "Length",
                   "Protein families", "Sequence"]

IUIS_RE = re.compile(r"[Aa]llergen\s+([A-Z][a-z]{1,3}\s+[a-z]{1,2}\s+\d+)")

# UniProt family strings for the pan-allergen sweep. A string that returns
# nothing anywhere is printed as such - it is then wrong, not absent.
PAN_FAMILIES = [
    "Profilin family",
    "Plant LTP family",
    "2S seed storage albumins family",
    "11S seed storage protein (globulins) family",
    "7S seed storage protein family",
    "BetVI family",
    "Thaumatin family",
    "Oleosin family",
    "Tropomyosin family",
    "Parvalbumin family",
]

# Sub-extract rules for panel entries that share an organism with another.
# Each rule is a list of substrings matched, case-insensitively, against the
# retrieved protein-name string. EXCLUDE wins over INCLUDE.
SUBEXTRACT = {
    "Egg White": dict(taxon="Gallus gallus", include=[
        "ovalbumin", "ovomucoid", "ovotransferrin", "conalbumin", "lysozyme",
        "ovoinhibitor", "ovoglobulin", "ovomacroglobulin", "cystatin",
        "ovostatin"], exclude=[]),
    "Egg Yolk": dict(taxon="Gallus gallus", include=[
        "vitellogenin", "livetin", "apovitellenin", "phosvitin",
        "vitelline"], exclude=[]),
    "Chicken": dict(taxon="Gallus gallus", include=[
        "parvalbumin", "enolase", "aldolase", "myoglobin", "myosin", "actin",
        "creatine kinase", "triosephosphate"], exclude=[]),
    "Casein (nBos d8)": dict(taxon="Bos taurus", include=["casein"],
                             exclude=["dander", "s100"]),
    "Cow Milk": dict(taxon="Bos taurus", include=[
        "casein", "lactoglobulin", "lactalbumin", "lactoferrin",
        "lactotransferrin", "albumin"], exclude=["dander", "s100"]),
    "Cow Milk, UHT": dict(taxon="Bos taurus", include=[
        "casein", "lactoglobulin", "lactalbumin", "lactoferrin",
        "lactotransferrin", "albumin"], exclude=["dander", "s100"]),
    "Meat": dict(taxon=None, include=[
        "myoglobin", "myosin", "actin", "tropomyosin", "parvalbumin",
        "enolase", "aldolase", "creatine kinase", "serum albumin"],
        exclude=["dander", "s100", "casein", "lactoglobulin", "lactalbumin"]),
    "Gluten": dict(taxon="Triticum aestivum",
                   include=["gliadin", "glutenin", "prolamin", "secalin"],
                   exclude=[]),
}
# --------------------------------------------------------------------------


def get(url, params, tries=3):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r
            print(f"    HTTP {r.status_code} (attempt {k+1})")
        except requests.RequestException as e:
            print(f"    {e} (attempt {k+1})")
        time.sleep(2 * (k + 1))
    return None


def query(q, size=500):
    r = get(KB_URL, {"query": q, "fields": FIELDS, "format": "tsv", "size": size})
    time.sleep(PAUSE)
    if r is None:
        return None
    lines = r.text.rstrip("\n").split("\n")
    if not lines or not lines[0]:
        return []
    header = lines[0].split("\t")
    if header != EXPECTED_HEADER:
        print(f"\nSTOP: unexpected UniProt columns.\n  expected {EXPECTED_HEADER}"
              f"\n  received {header}")
        sys.exit(1)
    out = []
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) == len(header):
            out.append(dict(zip(header, p)))
    return out


def iuis_codes(scientific_name):
    """Generate the WHO/IUIS source code from the organism name by the
    nomenclature rule itself. Both 3- and 4-letter genus prefixes and 1- and
    2-letter species prefixes are tried, because the official code lengthens
    only to break collisions (Sol t -> Sola t)."""
    parts = scientific_name.split()
    if len(parts) < 2:
        return []
    genus, species = parts[0], parts[1]
    codes = set()
    for g in (genus[:3], genus[:4]):
        for s in (species[:1], species[:2]):
            codes.add(f"{g.capitalize()} {s.lower()}")
    return sorted(codes)


def fetch_taxon(taxid, sci_name):
    """Tiered retrieval. Earlier tiers win; every record carries its tier."""
    seen, recs = set(), []

    def add(rows, tier):
        n = 0
        for r in rows or []:
            if r["Entry"] in seen:
                continue
            seen.add(r["Entry"])
            r["tier"] = tier
            recs.append(r)
            n += 1
        return n

    n1 = add(query(f"(taxonomy_id:{taxid}) AND (keyword:KW-0020) AND (reviewed:true)"),
             "1_reviewed_allergen")
    n2 = add(query(f"(taxonomy_id:{taxid}) AND (keyword:KW-0020)"),
             "2_any_allergen")
    n3 = 0
    if INCLUDE_TIER3:
        codes = iuis_codes(sci_name)
        if codes:
            sub = " OR ".join(f'protein_name:"{c}"' for c in codes)
            n3 = add(query(f"(taxonomy_id:{taxid}) AND ({sub})"), "3_iuis_name")
    n4 = 0
    if INCLUDE_TIER4 and not recs:
        fam = " OR ".join(f'family:"{f}"' for f in PAN_FAMILIES)
        n4 = add(query(f"(taxonomy_id:{taxid}) AND ({fam})"), "4_pan_family")
    return recs, (n1, n2, n3, n4)


def main():
    print("=" * 78)
    print("P2 - STAGE 3b: FINAL ALLERGEN SEQUENCE SETS")
    print("=" * 78)

    src = pd.read_csv(SOURCES_PATH)
    src["taxid"] = src["taxid"].astype("Int64")
    assert src["taxid"].notna().all(), "A source organism has no taxid"
    taxa = src.drop_duplicates("source_organism")[["source_organism", "taxid"]]
    print(f"\nPanel entries {src.panel_entry.nunique()}   "
          f"source organisms {len(taxa)}")

    store, dropped = {}, []
    print("\nTiered retrieval (1 reviewed+keyword | 2 any+keyword | "
          "3 WHO/IUIS name | 4 pan-allergen family)")
    for row in taxa.itertuples():
        recs, counts = fetch_taxon(int(row.taxid), row.source_organism)
        keep = []
        for r in recs:
            L = int(r["Length"])
            if L < MIN_LENGTH:
                dropped.append({"source_organism": row.source_organism,
                                "accession": r["Entry"], "length": L,
                                "reason": f"fragment < {MIN_LENGTH} aa",
                                "protein_names": r["Protein names"]})
                continue
            keep.append(r)
        store[row.source_organism] = keep
        print(f"  {row.source_organism:24s} t1 {counts[0]:3d}  t2 {counts[1]:3d}  "
              f"t3 {counts[2]:3d}  t4 {counts[3]:3d}  -> kept {len(keep):3d}")

    # ---- assemble per panel entry, applying sub-extract rules -------------
    rows, seqs = [], {}
    rule_hits = defaultdict(int)
    for entry, grp in src.groupby("panel_entry"):
        rule = SUBEXTRACT.get(entry)
        for org in grp.source_organism:
            for r in store.get(org, []):
                name = r["Protein names"].lower()
                if rule:
                    if rule["taxon"] and rule["taxon"] != org:
                        continue
                    if any(x in name for x in rule["exclude"]):
                        continue
                    if not any(x in name for x in rule["include"]):
                        continue
                    rule_hits[entry] += 1
                rows.append({
                    "panel_entry": entry, "source_organism": org,
                    "accession": r["Entry"], "iuis_name": "; ".join(sorted(
                        {re.sub(r"\s+", " ", m)
                         for m in IUIS_RE.findall(r["Protein names"])})),
                    "length": int(r["Length"]), "tier": r["tier"],
                    "families": r["Protein families"],
                    "protein_names": r["Protein names"]})
                seqs[r["Entry"]] = r["Sequence"]

    t = pd.DataFrame(rows)
    # exact-duplicate sequences inside one panel entry add nothing
    t["seq"] = t.accession.map(seqs)
    before = len(t)
    t = t.drop_duplicates(["panel_entry", "seq"]).drop(columns="seq")
    print(f"\nIdentical sequences removed within panel entries: {before - len(t)}")

    for entry, rule in SUBEXTRACT.items():
        if rule_hits[entry] == 0:
            print(f"\nSTOP: sub-extract rule for {entry!r} matched no protein. "
                  f"The rule is wrong, or retrieval returned nothing for "
                  f"{rule['taxon']}.")
            sys.exit(1)

    t.to_csv(OUT_SETS, index=False)
    pd.DataFrame(dropped).to_csv(OUT_DROPPED, index=False)
    with open(OUT_FASTA, "w") as fh:
        for acc in sorted(set(t.accession)):
            fh.write(f">{acc}\n")
            s = seqs[acc]
            for i in range(0, len(s), 60):
                fh.write(s[i:i+60] + "\n")

    print("\n=== FINAL SET PER PANEL ENTRY ===")
    empty = []
    for entry in sorted(src.panel_entry.unique()):
        sub = t[t.panel_entry == entry]
        if not len(sub):
            empty.append(entry)
        iuis = sorted({i for v in sub.iuis_name for i in str(v).split("; ") if i})
        tiers = sub.tier.value_counts().to_dict() if len(sub) else {}
        tag = "  <-- EMPTY" if not len(sub) else ""
        print(f"  {entry:18s} {len(sub):3d} proteins  {len(iuis):2d} named  "
              f"{ {k.split('_')[0]: v for k, v in tiers.items()} }  "
              f"{', '.join(iuis[:6])}{' ...' if len(iuis) > 6 else ''}{tag}")

    print("\n=== REMAINING DEGENERACY ===")
    sig = defaultdict(list)
    for entry in src.panel_entry.unique():
        key = tuple(sorted(t[t.panel_entry == entry].accession))
        if key:
            sig[key].append(entry)
    any_deg = False
    for key, group in sig.items():
        if len(group) > 1:
            any_deg = True
            print(f"  identical sets: {group} ({len(key)} proteins)")
    for a, b in [("Casein (nBos d8)", "Cow Milk"), ("Gluten", "Wheat Flour")]:
        sa = set(t[t.panel_entry == a].accession)
        sb = set(t[t.panel_entry == b].accession)
        if sa and sb and (sa <= sb or sb <= sa):
            any_deg = True
            print(f"  subset relation: {a} ({len(sa)}) within {b} ({len(sb)})")
    if not any_deg:
        print("  none")

    print("\n=== FRAGMENTS EXCLUDED ===")
    if dropped:
        for d0 in dropped:
            print(f"  {d0['accession']:10s} {d0['length']:4d} aa  "
                  f"{d0['source_organism']:22s} {d0['protein_names'][:52]}")
    else:
        print("  none")

    print("\n=== STATUS ===")
    print(f"  panel entries with sequences: {src.panel_entry.nunique() - len(empty)}"
          f" / {src.panel_entry.nunique()}")
    print(f"  still empty: {empty}")
    print(f"  unique proteins: {t.accession.nunique()}   rows: {len(t)}")
    print(f"\nWritten: {OUT_SETS}, {OUT_FASTA}, {OUT_DROPPED}")
    print("=" * 78)


if __name__ == "__main__":
    main()
