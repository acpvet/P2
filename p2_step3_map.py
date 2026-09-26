"""
P2 - Stage 3: map the 34 panel entries to allergen sequence sets.

Design decision taken at Stage 2: extract-based testing is represented by a
SET of molecular allergens per panel entry, not a single representative. Every
later similarity metric between two panel entries is therefore computed over
all allergen x allergen combinations and aggregated (max, and mean of top-k).

Nothing about the allergens themselves is typed here. The only hand-entered
content is the SOURCE ORGANISM of each panel entry - a taxonomic claim that
this script verifies against the UniProt taxonomy service and prints for
inspection. WHO/IUIS designations (Ses i 5, Ara h 2, ...), accessions,
lengths, families and sequences are all retrieved from UniProt and parsed out
of the returned records.

What this stage must expose, not hide:
  - panel entries with no reviewed allergen in UniProt (they cannot enter the
    computational layer as designed);
  - panel entries that resolve to the SAME organism (Cow Milk / Cow Milk UHT /
    Casein; Egg White / Egg Yolk / Chicken; Gluten / Wheat Flour). At organism
    level their sequence sets are identical, which would force a similarity of
    1.0 and corrupt the linkage. Stage 3b resolves these by allergen-level
    subsetting; this stage only measures how large the problem is.

Run:
    python p2_step3_map.py

Requires: pandas, requests.   Needs internet. Runtime ~2 min.
"""

import os
import re
import sys
import time
from collections import defaultdict

import pandas as pd
import requests

# --------------------------------------------------------------------------
# The only curated content in this file: panel entry -> source organism(s).
# Names must be UniProt scientific names; unresolved ones are reported, not
# silently dropped.
# --------------------------------------------------------------------------
PANEL_SOURCES = {
    "Apple":            ["Malus domestica"],
    "Banana":           ["Musa acuminata"],
    "Carrot":           ["Daucus carota"],
    "Casein (nBos d8)": ["Bos taurus"],
    "Cherry":           ["Prunus avium"],
    "Chicken":          ["Gallus gallus"],
    "Cow Milk":         ["Bos taurus"],
    "Cow Milk, UHT":    ["Bos taurus"],
    "Egg White":        ["Gallus gallus"],
    "Egg Yolk":         ["Gallus gallus"],
    "Fig":              ["Ficus carica"],
    "Fish":             ["Gadus morhua", "Salmo salar",
                         "Oreochromis niloticus", "Thunnus albacares"],
    "Gluten":           ["Triticum aestivum"],
    "Hazelnut":         ["Corylus avellana"],
    "Kiwi":             ["Actinidia deliciosa"],
    "Meat":             ["Bos taurus", "Ovis aries"],
    "Mulberry":         ["Morus alba", "Morus nigra"],
    "Olive":            ["Olea europaea"],
    "Onion":            ["Allium cepa"],
    "Orange":           ["Citrus sinensis"],
    "Peach":            ["Prunus persica"],
    "Peanut":           ["Arachis hypogaea"],
    "Pistachio":        ["Pistacia vera"],
    "Potato":           ["Solanum tuberosum"],
    "Pumpkin Seeds":    ["Cucurbita pepo", "Cucurbita maxima"],
    "Rice":             ["Oryza sativa"],
    "Sesame":           ["Sesamum indicum"],
    "Shellifish":       ["Penaeus monodon", "Penaeus vannamei",
                         "Homarus americanus"],
    "Soybean":          ["Glycine max"],
    "Strawberry":       ["Fragaria ananassa"],
    "String Bean":      ["Phaseolus vulgaris"],
    "Sunflower Seeds":  ["Helianthus annuus"],
    "Tomato":           ["Solanum lycopersicum"],
    "Wheat Flour":      ["Triticum aestivum"],
}

MATRIX_PATH = "p2_binary_matrix.csv"   # to assert the panel matches the data
INCLUDE_UNREVIEWED = False             # reviewed (Swiss-Prot) only by default
PAUSE = 0.3                            # seconds between API calls
OUT_TABLE = "p2_allergen_candidates.csv"
OUT_FASTA = "p2_sequences.fasta"
OUT_SOURCES = "p2_panel_sources.csv"

TAX_URL = "https://rest.uniprot.org/taxonomy/search"
KB_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,protein_name,organism_name,length,cc_allergen,protein_families,sequence"
EXPECTED_HEADER = ["Entry", "Protein names", "Organism", "Length",
                   "Allergenic Properties", "Protein families", "Sequence"]

# WHO/IUIS designation as it appears inside the protein-name string,
# e.g. "(Allergen Ses i 5)" or "(allergen Ara h 2.0101)".
IUIS_RE = re.compile(r"[Aa]llergen\s+([A-Z][a-z]{2}\s+[a-z]{1,2}\s+\d+)")
# --------------------------------------------------------------------------


def get(url, params, tries=3):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            print(f"    HTTP {r.status_code} (attempt {k+1})")
        except requests.RequestException as e:
            print(f"    request failed: {e} (attempt {k+1})")
        time.sleep(2 * (k + 1))
    return None


def check_panel_matches_data():
    if not os.path.exists(MATRIX_PATH):
        print(f"NOTE: {MATRIX_PATH} not found - skipping the panel check.")
        return
    cols = [c.strip() for c in pd.read_csv(MATRIX_PATH, nrows=0).columns][1:]
    only_data = sorted(set(cols) - set(PANEL_SOURCES))
    only_here = sorted(set(PANEL_SOURCES) - set(cols))
    if only_data or only_here:
        print("PANEL MISMATCH between PANEL_SOURCES and the binary matrix:")
        print(f"  in the data but not mapped : {only_data}")
        print(f"  mapped but not in the data : {only_here}")
        sys.exit(1)
    print(f"Panel check: all {len(cols)} allergens in the data are mapped.")


def resolve_taxon(name):
    r = get(TAX_URL, {"query": f'scientific:"{name}"', "format": "json",
                      "size": 5})
    if r is None:
        return None, "request failed"
    hits = r.json().get("results", [])
    if not hits:
        return None, "no taxonomy hit"
    exact = [h for h in hits if h.get("scientificName", "").lower() == name.lower()]
    h = exact[0] if exact else hits[0]
    note = "" if exact else f"inexact -> {h.get('scientificName')}"
    return h, note


def fetch_allergens(taxid):
    q = f"(taxonomy_id:{taxid}) AND (keyword:KW-0020)"
    if not INCLUDE_UNREVIEWED:
        q += " AND (reviewed:true)"
    r = get(KB_URL, {"query": q, "fields": FIELDS, "format": "tsv",
                     "size": 500})
    if r is None:
        return None, "request failed"
    lines = r.text.rstrip("\n").split("\n")
    if not lines or not lines[0]:
        return [], "empty response"
    header = lines[0].split("\t")
    if header != EXPECTED_HEADER:
        print("\nSTOP: UniProt returned unexpected columns.")
        print(f"  expected: {EXPECTED_HEADER}")
        print(f"  received: {header}")
        sys.exit(1)
    out = []
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) != len(header):
            continue
        rec = dict(zip(header, p))
        names = IUIS_RE.findall(rec["Protein names"])
        canon = sorted({re.sub(r"\s+", " ", n) for n in names})
        rec["iuis"] = "; ".join(canon)
        out.append(rec)
    return out, ""


def main():
    print("=" * 78)
    print("P2 - STAGE 3: PANEL ENTRY -> ALLERGEN SEQUENCE SETS")
    print("=" * 78)
    check_panel_matches_data()

    names = sorted({n for v in PANEL_SOURCES.values() for n in v})
    print(f"\nResolving {len(names)} source organisms against UniProt taxonomy")
    taxa, unresolved = {}, []
    for n in names:
        h, note = resolve_taxon(n)
        time.sleep(PAUSE)
        if h is None:
            unresolved.append((n, note))
            print(f"  {n:26s} UNRESOLVED ({note})")
            continue
        taxa[n] = h
        lineage = [x["scientificName"] for x in h.get("lineage", [])
                   if x.get("rank") in ("family", "order")]
        print(f"  {n:26s} taxid {h['taxonId']:<8} {h.get('rank',''):8s} "
              f"{' / '.join(lineage)} {note}")

    print(f"\nFetching allergen records "
          f"({'reviewed + unreviewed' if INCLUDE_UNREVIEWED else 'reviewed only'})")
    by_taxon = {}
    for n, h in taxa.items():
        recs, err = fetch_allergens(h["taxonId"])
        time.sleep(PAUSE)
        if recs is None:
            print(f"  {n:26s} FAILED ({err})")
            by_taxon[n] = []
            continue
        by_taxon[n] = recs
        iuis = sorted({i for r in recs for i in r["iuis"].split("; ") if i})
        print(f"  {n:26s} {len(recs):3d} entries   {', '.join(iuis) if iuis else '(no WHO/IUIS name parsed)'}")

    rows, seqs = [], {}
    for entry, sources in PANEL_SOURCES.items():
        for src in sources:
            for r in by_taxon.get(src, []):
                rows.append({
                    "panel_entry": entry, "source_organism": src,
                    "taxid": taxa[src]["taxonId"],
                    "accession": r["Entry"], "iuis_name": r["iuis"],
                    "organism": r["Organism"], "length": int(r["Length"]),
                    "families": r["Protein families"],
                    "protein_names": r["Protein names"],
                })
                seqs[r["Entry"]] = (r["Sequence"], r["iuis"], r["Organism"])

    t = pd.DataFrame(rows)
    t.to_csv(OUT_TABLE, index=False)
    with open(OUT_FASTA, "w") as fh:
        for acc, (seq, iuis, org) in sorted(seqs.items()):
            fh.write(f">{acc}|{iuis or 'NA'}|{org}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")

    pd.DataFrame([{"panel_entry": e, "source_organism": s,
                   "taxid": taxa[s]["taxonId"] if s in taxa else None}
                  for e, v in PANEL_SOURCES.items() for s in v]
                 ).to_csv(OUT_SOURCES, index=False)

    print("\n=== COVERAGE PER PANEL ENTRY ===")
    empty = []
    for entry in PANEL_SOURCES:
        sub = t[t.panel_entry == entry]
        iuis = sorted({i for v in sub.iuis_name for i in str(v).split("; ") if i})
        flag = "" if len(sub) else "   <-- NO SEQUENCES"
        if not len(sub):
            empty.append(entry)
        print(f"  {entry:18s} {len(sub):3d} proteins  "
              f"{len(iuis):2d} WHO/IUIS  {', '.join(iuis[:8])}"
              f"{' ...' if len(iuis) > 8 else ''}{flag}")

    print("\n=== DEGENERATE ENTRIES (identical sequence set) ===")
    sig = defaultdict(list)
    for entry in PANEL_SOURCES:
        key = tuple(sorted(t[t.panel_entry == entry].accession))
        if key:
            sig[key].append(entry)
    n_deg = 0
    for key, group in sig.items():
        if len(group) > 1:
            n_deg += 1
            print(f"  {group}  -> {len(key)} shared accessions; "
                  f"{len(group)*(len(group)-1)//2} pair(s) would score "
                  f"similarity 1.0")
    if not n_deg:
        print("  none")

    print("\n=== WHAT BLOCKS THE COMPUTATIONAL LAYER ===")
    if unresolved:
        print(f"  unresolved organisms ({len(unresolved)}): "
              f"{[u[0] for u in unresolved]}")
    print(f"  panel entries with no sequence ({len(empty)}): {empty}")
    print(f"  unique proteins retrieved: {len(seqs)}")
    print(f"  total panel-entry x protein rows: {len(t)}")

    print(f"\nWritten: {OUT_TABLE}, {OUT_FASTA}, {OUT_SOURCES}")
    print("=" * 78)


if __name__ == "__main__":
    main()
