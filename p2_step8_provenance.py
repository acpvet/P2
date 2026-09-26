"""
P2 - Stage 8: provenance for the Methods section.

Two things in the Methods cannot be written from memory: the UniProt release
the sequences came from, and the version of every package that produced a
number. Both are read from the running system here and written out as text
ready to paste.

The UniProt release is returned in the response headers of any REST call. The
header name has changed before, so this asks for one tiny record and prints
every header whose name mentions a release or a date, then falls back to
printing all headers if none match - it never guesses a release number.

Run:
    python p2_step8_provenance.py

Requires: requests. Everything else is standard library.
"""

import platform
import sys
from importlib import metadata

import requests

# --------------------------------------------------------------------------
PACKAGES = ["numpy", "scipy", "pandas", "biopython", "requests",
            "transformers", "torch", "matplotlib", "networkx", "openpyxl",
            "huggingface-hub", "safetensors"]
IMPORT_NAME = {"biopython": "Bio", "huggingface-hub": "huggingface_hub"}
CACHE = "p2_esm2_embeddings.npz"
PROBE = ("https://rest.uniprot.org/uniprotkb/search"
         "?query=accession:P02662&fields=accession&format=tsv&size=1")
OUT = "p2_provenance.md"
# --------------------------------------------------------------------------


def uniprot_release():
    print("Querying UniProt for its release headers ...")
    try:
        r = requests.get(PROBE, timeout=60)
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return None, None, {}
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}")
        return None, None, dict(r.headers)
    hits = {k: v for k, v in r.headers.items()
            if "release" in k.lower() or "date" in k.lower()}
    if not hits:
        print("  No header mentions a release. All headers returned:")
        for k, v in r.headers.items():
            print(f"    {k}: {v}")
        return None, None, dict(r.headers)
    for k, v in hits.items():
        print(f"  {k}: {v}")
    rel = next((v for k, v in hits.items()
                if "release" in k.lower() and "date" not in k.lower()), None)
    # the release date header is the one naming BOTH; a bare "date" header is
    # the HTTP date or the API deployment date, which is a different thing
    date = next((v for k, v in hits.items()
                 if "release" in k.lower() and "date" in k.lower()), None)
    return rel, date, dict(r.headers)


def versions():
    rows = []
    for p in PACKAGES:
        try:
            rows.append((p, metadata.version(p)))
        except metadata.PackageNotFoundError:
            mod = IMPORT_NAME.get(p, p)
            try:
                rows.append((p, metadata.version(mod)))
            except Exception:
                rows.append((p, "not installed"))
    return rows


def esm_model():
    try:
        import numpy as np
        z = np.load(CACHE, allow_pickle=True)
        return str(z["model"]), int(z["E"].shape[0]), int(z["E"].shape[1])
    except Exception as e:
        return f"(could not read {CACHE}: {e})", None, None


def main():
    print("=" * 74)
    print("P2 - STAGE 8: PROVENANCE")
    print("=" * 74)

    rel, date, headers = uniprot_release()
    vers = versions()
    model, n_prot, dim = esm_model()

    py = sys.version.split()[0]
    osname = f"{platform.system()} {platform.release()} ({platform.machine()})"

    print("\n=== ENVIRONMENT ===")
    print(f"  OS      {osname}")
    print(f"  Python  {py}")
    print(f"  ESM-2   {model}" + (f"   {n_prot} x {dim}" if n_prot else ""))
    print("\n=== PACKAGE VERSIONS ===")
    for p, v in vers:
        print(f"  {p:18s} {v}")

    rel_txt = (f"UniProt release {rel}" if rel else
               "UniProt release «not returned - see the header list above»")
    date_txt = f" (released {date})" if date else ""

    lines = [
        "# P2 - provenance for Methods §2.8",
        "",
        "## Sentence to paste into Methods",
        "",
        f"Sequences were retrieved from UniProtKB ({rel_txt}{date_txt}), "
        f"accessed 25 September 2026. Analyses were run under "
        f"{osname} with Python {py}. Protein language model embeddings used "
        f"{model}. Package versions are listed in Supplementary Table S7, and "
        "all random seeds were fixed at 20260925.",
        "",
        "## Supplementary Table S7 - software versions",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    lines += [f"| {p} | {v} |" for p, v in vers]
    lines += [f"| Python | {py} |", f"| OS | {osname} |", "",
              "## UniProt response headers (verbatim)", "", "```"]
    lines += [f"{k}: {v}" for k, v in headers.items()] or ["(none returned)"]
    lines += ["```", ""]

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nWritten: {OUT}")
    print("=" * 74)


if __name__ == "__main__":
    main()
