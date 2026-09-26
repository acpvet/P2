"""
P2 - Stage 11: de-identify the binary matrix before deposition.

The patient identifiers carried through from the laboratory file are not
sequential study numbers. They range from 1 to 140752, and one of them is a
laboratory accession string ('124182I'). Identifiers of that kind can be
linked back to laboratory records, so they must not appear in a public
deposit even though the file contains no name, date or demographic field.

This replaces every identifier with a sequential study number, in an order
fixed by a seeded shuffle so that the published row order carries no
information about the original numbering. The mapping is written to a
separate file that stays OUT of the deposit and out of the repository; keep it
with the study records so a reviewer query can still be traced back.

Run:
    python p2_step11_deidentify.py
"""

import numpy as np
import pandas as pd

IN = "p2_binary_matrix.csv"
OUT = "p2_binary_matrix_deid.csv"
MAP = "p2_id_map_PRIVATE_DO_NOT_DEPOSIT.csv"
SEED = 20260925


def main():
    print("=" * 74)
    print("P2 - STAGE 11: DE-IDENTIFY FOR DEPOSITION")
    print("=" * 74)

    d = pd.read_csv(IN, index_col=0)
    ids = [str(i) for i in d.index]
    assert len(set(ids)) == len(ids), "duplicate identifiers"

    nonnum = [i for i in ids if not i.isdigit()]
    nums = [int(i) for i in ids if i.isdigit()]
    print(f"\nRows {len(ids)}   columns {d.shape[1]}")
    print(f"  numeric identifiers: {len(nums)}  range {min(nums)}–{max(nums)}")
    print(f"  non-numeric identifiers: {nonnum if nonnum else 'none'}")
    print("  These are laboratory identifiers, not study numbers.")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ids))
    new = [f"P{i + 1:03d}" for i in range(len(ids))]
    mapping = {ids[old]: new[k] for k, old in enumerate(order)}

    out = d.copy()
    out.index = [mapping[str(i)] for i in d.index]
    out.index.name = "study_id"
    out = out.sort_index()
    out.to_csv(OUT)

    pd.DataFrame({"original_identifier": list(mapping.keys()),
                  "study_id": list(mapping.values())}).to_csv(MAP, index=False)

    # the deposited file must carry no trace of the original numbering
    assert not set(out.index) & set(ids), "an original identifier survived"
    assert out.shape == d.shape, "shape changed"
    assert int(out.values.sum()) == int(d.values.sum()), "positives changed"
    recovered = out.loc[sorted(out.index)].values.sum(axis=0)
    assert (recovered == d.values.sum(axis=0)).all(), "column totals changed"

    print(f"\n  written {OUT}  (study_id P001–P{len(ids):03d}, row order shuffled)")
    print(f"  written {MAP}")
    print("\n  KEEP THE MAP OUT OF THE DEPOSIT AND OUT OF THE REPOSITORY.")
    print("  It is the only link back to the laboratory records.")
    print("=" * 74)


if __name__ == "__main__":
    main()
