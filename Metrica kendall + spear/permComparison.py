import csv
from pathlib import Path
from itertools import combinations

from load_permutations import load_all_permutations
from kendallSpearman import (
    truncated_rank,
)

# -------------------------
# Configuration parameters
# -------------------------

# Cutoff: number of elements to consider for Spearman
L_SPEARMAN = 20
GAMMA = 1.0

OUTPUT_CSV = "perm_distances.csv"


def save_results_to_csv(rows, output_path: str) -> None:
    """
    Save all distance results to a CSV file.

    Each element in `rows` must be a dict with keys:
    - id
    - method_a
    - method_b
    - spearman_raw
    - spearman_norm
    """
    fieldnames = [
        "id",
        "method_a",
        "method_b",
        "spearman_raw",
        "spearman_norm",
    ]

    out_path = Path(output_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved {len(rows)} rows to: {out_path.resolve()}")


def spearman_cutoff_penalized(A, B, cutoff, gamma):
    """
    Compute Spearman distance on the union of the top-'cutoff' elements,
    with ranks truncated at 'cutoff'.
    """
    if len(A) != len(B):
        raise ValueError("Le permutazioni A e B hanno dimensione diversa.")
    if set(A) != set(B):
        raise ValueError("A e B non contengono lo stesso insieme di elementi.")
    if len(set(A)) != len(A):
        raise ValueError("La permutazione A contiene elementi duplicati.")
    if len(set(B)) != len(B):
        raise ValueError("La permutazione B contiene elementi duplicati.")
    if not isinstance(cutoff, int):
        raise ValueError("cutoff non è un intero.")
    if not (1 <= cutoff <= len(A)):
        raise ValueError("cutoff è fuori dall'intervallo consentito.")
    if not isinstance(gamma, (int, float)):
        raise ValueError("gamma non è numerico.")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma è fuori dall'intervallo [0, 1].")

    A_cut = set(A[:cutoff])
    B_cut = set(B[:cutoff])

    U = A_cut | B_cut
    common_cut = A_cut & B_cut

    total = 0.0
    for x in U:
        w = float(gamma) if x in common_cut else 1.0
        S_A = truncated_rank(x, A, cutoff)
        S_B = truncated_rank(x, B, cutoff)
        total += w * abs(S_A - S_B)

    return total


def normalize_spearman_cutoff(distance: float, cutoff: int) -> float:
    """
    Normalize a Spearman cutoff distance using the theoretical maximum.
    """
    if not isinstance(distance, (int, float)):
        raise ValueError("distance must be numeric.")
    if distance < 0:
        raise ValueError("distance cannot be negative.")
    if not isinstance(cutoff, int):
        raise ValueError("cutoff must be an integer.")
    if cutoff <= 1:
        raise ValueError("cutoff must be greater than 1.")

    max_distance = cutoff * (cutoff - 1)
    return float(distance) / max_distance


if __name__ == "__main__":
    base_dir = "./vicini"

    permutations = load_all_permutations(base_dir)
    print(f"Loaded {len(permutations)} IDs from directory: {base_dir}\n")

    methods = [
        "bert",
        "glove",
        "w2v",
        "fasttext",
        "roberta",
        "clip",
        "gpt-2",
        "minilm",
        "mistral",
        "mpnet",
        "qwen25",
        "sentence_bert",
        "t5",
        "xlnet",
    ] 

    total_ids = len(permutations)
    pairs_per_id = len(methods) * (len(methods) - 1) // 2
    ids_with_all_methods = [
        id_ for id_, method_perms in permutations.items()
        if all(m in method_perms for m in methods)
    ]
    total_valid_ids = len(ids_with_all_methods)
    total_comparisons = total_valid_ids * pairs_per_id

    print(f"Methods: {len(methods)} | pairs per ID: {pairs_per_id}")
    print(f"IDs total: {total_ids} | IDs with all methods: {total_valid_ids}")
    print(f"Total comparisons planned: {total_comparisons}\n")

    # This list will collect all results to be written to CSV
    all_rows = []
    comparison_idx = 0
    valid_id_idx = 0

    for idx, (id_, method_perms) in enumerate(permutations.items(), start=1):
        print(f"========== ID {id_} ({idx} of {len(permutations)}) ==========\n")

        # Check that all methods are available for this ID
        missing_methods = [m for m in methods if m not in method_perms]
        if missing_methods:
            print(f"Skipping ID {id_}: missing methods {missing_methods}")
            continue
        valid_id_idx += 1
        print(
            f"Starting comparisons for ID {id_} "
            f"({valid_id_idx} of {total_valid_ids})"
        )

        # Iterate over all unordered pairs of methods
        for method_a, method_b in combinations(methods, 2):
            comparison_idx += 1
            print(
                f"Progress: comparison {comparison_idx} "
                f"of {total_comparisons}"
            )
            perm_a = method_perms[method_a]
            perm_b = method_perms[method_b]

            # Spearman
            spearman_raw = spearman_cutoff_penalized(
                perm_a, perm_b, L_SPEARMAN, GAMMA
            )
            spearman_norm = normalize_spearman_cutoff(spearman_raw, L_SPEARMAN)

            # Print to console
            print(f"Comparison: ID {id_} | {method_a} vs {method_b}")
            print(
                f"  Spearman_cutoff (cutoff={L_SPEARMAN}, gamma={GAMMA}):"
            )
            print(f"    raw        = {spearman_raw:.6f}")
            print(f"    normalized = {spearman_norm:.6f}")
            print()

            # Store result for CSV
            all_rows.append(
                {
                    "id": id_,
                    "method_a": method_a,
                    "method_b": method_b,
                    "spearman_raw": spearman_raw,
                    "spearman_norm": spearman_norm,
                }
            )
        print(
            f"Completed ID {id_} "
            f"({valid_id_idx} of {total_valid_ids})\n"
        )

    # Save everything to CSV at the end
    save_results_to_csv(all_rows, OUTPUT_CSV)
    print("All done.")
