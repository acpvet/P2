# Sequence similarity and clinical co-sensitization in a Jordanian food allergy cohort

Analysis code and derived data for the manuscript:

> Al Athamneh, A., Khaleel, A. and Alshatali, S. Protein family membership
> outperforms sequence identity and a protein language model in predicting food
> allergen co-sensitization: evidence from a Jordanian cohort. «journal, year».

The question: which computational measure of allergen similarity matches
observed co-sensitization, once the confounders on each side are removed?
Clinical co-occurrence is adjusted against a null holding each patient's
degree of sensitization and each allergen's prevalence fixed; each
computational metric is adjusted against a null holding the number of
candidate proteins per food fixed.

## Pipeline

Each stage reads the previous stage's output file and asserts its shape before
proceeding. Run them in order from this directory.

| Stage | Script | Produces |
|---|---|---|
| 1b | `p2_step1b_reconstruct.py` | pairwise 2×2 tables reconstructed from the published summary (cross-check only) |
| 2 | `p2_step2_adjusted.py` | `p2_binary_matrix.csv`, `p2_pair_table_adjusted.csv` — curveball-adjusted co-sensitization |
| 11 | `p2_step11_deidentify.py` | `p2_binary_matrix_deid.csv` — study numbers in place of laboratory identifiers, for deposition |
| 3 | `p2_step3_map.py` | `p2_allergen_candidates.csv`, `p2_panel_sources.csv` — UniProt retrieval |
| 3b | `p2_step3b_refine.py` | `p2_allergen_sets.csv`, `p2_dropped_records.csv` — tiered retrieval, fragment removal |
| 3c | `p2_step3c_collapse.py` | `p2_allergen_sets_final.csv`, `p2_sequences_curated.fasta` — one representative per allergenic family |
| 4 | `p2_step4_metrics.py` | `p2_protein_pair_metrics.csv`, `p2_metrics_pairs.csv` — identity, FAO/WHO window, k-mers, family |
| 5 | `p2_step5_linkage.py` | `p2_linkage_pairs.csv`, `p2_linkage_summary.csv` — set-size-preserving null, linkage |
| 6 | `p2_step6_esm2.py` | `p2_esm2_pairs.csv`, `p2_esm2_embeddings.npz` — ESM-2 embeddings |
| 7 | `p2_step7_figures.py` | `figures/` — Figures 1–5, Tables 1–2 |
| 8 | `p2_step8_provenance.py` | `p2_provenance.md` — UniProt release and package versions |
| 9 | `p2_step9_numbers.py` | `p2_numbers.json` — every value quoted in the manuscript |
| 10 | `p2_step10_manuscript.js` | `P2_manuscript.docx` (Node; `npm install docx`) |

Stages 3 and 3b need internet (UniProt REST). Stage 6 downloads the ESM-2
weights on first run (~2.5 GB) and caches the embeddings. Stage 4 takes about
17 minutes; stages 2, 5 and 6 about 4 minutes each.

All random seeds are fixed at 20260925. Re-running the pipeline reproduces
every number in the manuscript.

## Input data

`cleaned_allergy_15_sept.xlsx` is the individual-level clinical file and is
**not** included here. «State which of the two applies and delete the other:
(a) it is deposited under restricted access and available on request to the
corresponding author; (b) it is included as `data/` under the terms stated in
the manuscript.» The derived binary matrix and every aggregate table needed to
reproduce the analysis from stage 2 onward are included.

Sequence data are from UniProtKB release 2026_03 (released 2 September 2026,
accessed 25 September 2026) and are redistributed here under UniProt's
CC BY 4.0 terms.

## Environment

Windows 11, Python 3.12.10. See `requirements.txt` for pinned versions; the
recorded environment is in `p2_provenance.md`.

## Licence

Code: «MIT — see LICENSE; change if you prefer another».
Derived data and figures: CC BY 4.0.

