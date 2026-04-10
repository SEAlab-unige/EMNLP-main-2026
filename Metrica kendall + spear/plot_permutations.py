import os
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = "perm_distances.csv"
OUTPUT_DIR = "plots"
# Ordine fisso dei metodi per stampa/plot
METHOD_ORDER = [
    "w2v",
    "fasttext",
    "glove",
    "roberta",
    "sentence_bert",
    "gpt-2",
    "mistral",
    "qwen25",
    "bert",
    "mpnet",
    "minilm",
    "t5",
    "xlnet",
    "clip",
]
# Alias for heatmaps (used to order labels); same order as METHOD_ORDER
HEATMAP_METHOD_ORDER = METHOD_ORDER


def ensure_output_dir(path: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_results(csv_path: str) -> pd.DataFrame:
    """Load permutation distance results from CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def get_all_methods(df: pd.DataFrame):
    """Return the sorted list of all methods appearing in the CSV."""
    methods_a = df["method_a"].unique()
    methods_b = df["method_b"].unique()
    methods = sorted(set(methods_a) | set(methods_b))
    return methods


# -------------------- HEATMAPS --------------------


def build_distance_matrix(df: pd.DataFrame, metric: str, methods):
    """
    Build a symmetric distance matrix (methods x methods)
    using the mean of `metric` over all IDs for each pair of methods.
    """
    n = len(methods)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                matrix[i][j] = 0.0
            else:
                # Filter rows where the pair (m1, m2) appears in any order
                cond = (
                    ((df["method_a"] == m1) & (df["method_b"] == m2))
                    | ((df["method_a"] == m2) & (df["method_b"] == m1))
                )
                sub = df[cond]
                if len(sub) == 0:
                    value = 0.0  # fallback, should not happen with complete data
                else:
                    value = sub[metric].mean()
                matrix[i][j] = value

    return matrix


def plot_heatmap(matrix, methods, metric_name: str, output_dir: str):
    """Plot a heatmap with the numeric value printed INSIDE each cell."""

    matrix = np.array(matrix, dtype=float)
    n = len(methods)

    # Scale figure size with number of methods so labels/values stay readable
    width = max(8, n * 0.75)
    height = max(6, n * 0.6)
    plt.figure(figsize=(width, height))

    # Heatmap: blue (low) → red (high)
    img = plt.imshow(matrix, interpolation="nearest", cmap="RdBu_r")

    # Add colorbar
    cbar = plt.colorbar(img)
    cbar.set_label(f"{metric_name} (normalized)")

    # Tick labels
    plt.xticks(range(n), methods, rotation=45, ha="right")
    plt.yticks(range(n), methods)

    # Title
    plt.title(f"Average {metric_name} distance between methods")

    # ---- Print value INSIDE each cell ----
    mean_val = np.mean(matrix)
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]

            # Choose text color based on background intensity
            text_color = "white" if value > mean_val else "black"

            plt.text(
                j,               # x coordinate (column)
                i,               # y coordinate (row)
                f"{value:.3f}",  # formatted text
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold"
            )

    plt.tight_layout()

    # Save
    out_path = os.path.join(output_dir, f"heatmap_{metric_name}.png")
    plt.savefig(out_path, dpi=250)
    plt.close()

    print(f"Saved heatmap to {out_path}")


def generate_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Generate heatmaps for Kendall, Spearman and combined normalized distances.
    """
    methods = get_all_methods(df)
    methods = [
        m for m in HEATMAP_METHOD_ORDER if m in methods
    ] + [m for m in methods if m not in HEATMAP_METHOD_ORDER]

    for metric in ["kendall_norm", "spearman_norm", "combined_norm"]:
        matrix = build_distance_matrix(df, metric, methods)
        plot_heatmap(matrix, methods, metric, output_dir)


# -------------------- SCATTER Kendall vs Spearman --------------------
# (lasciamo solo il confronto Kendall vs Spearman, come prima)


def generate_scatter_plots(df: pd.DataFrame, output_dir: str):
    """
    For each pair of methods, create a scatter plot:
    x = Kendall_norm, y = Spearman_norm, one point per ID.
    """
    methods = get_all_methods(df)

    for m1, m2 in combinations(methods, 2):
        cond = (
            ((df["method_a"] == m1) & (df["method_b"] == m2))
            | ((df["method_a"] == m2) & (df["method_b"] == m1))
        )
        sub = df[cond]

        if len(sub) == 0:
            continue

        plt.figure(figsize=(5, 4))
        plt.scatter(
            sub["kendall_norm"],
            sub["spearman_norm"],
            alpha=0.7,
        )
        for _, row in sub.iterrows():
            plt.annotate(
                str(row["id"]),
                (row["kendall_norm"], row["spearman_norm"]),
                fontsize=6,
                alpha=0.7,
            )

        plt.xlabel("Kendall (normalized)")
        plt.ylabel("Spearman (normalized)")
        plt.title(f"Kendall vs Spearman | {m1} vs {m2}")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(alpha=0.3)

        fname = f"scatter_kendall_vs_spearman_{m1}_{m2}.png"
        out_path = os.path.join(output_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved scatter plot to {out_path}")


# -------------------- FOCUS PER ID --------------------


def generate_per_id_plots(df: pd.DataFrame, output_dir: str):
    """
    For each ID, create bar plots of distances for all method pairs.

    We create three separate plots per ID:
    - one for Kendall_norm
    - one for Spearman_norm
    - one for combined_norm
    """
    unique_ids = sorted(df["id"].unique())

    for id_ in unique_ids:
        sub = df[df["id"] == id_].copy()

        # Build labels like "bert-fasttext", "glove-w2v", ...
        pair_labels = [
            f"{a}-{b}" for a, b in zip(sub["method_a"], sub["method_b"])
        ]

        # --- Kendall ---
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(sub)), sub["kendall_norm"])
        plt.xticks(range(len(sub)), pair_labels, rotation=45, ha="right")
        plt.ylim(0.0, 1.05)
        plt.ylabel("Kendall (normalized)")
        plt.title(f"Kendall distances for ID {id_}")
        plt.tight_layout()

        fname_k = f"per_id_kendall_{id_}.png"
        out_path_k = os.path.join(output_dir, fname_k)
        plt.savefig(out_path_k, dpi=200)
        plt.close()
        print(f"Saved per-ID Kendall plot to {out_path_k}")

        # --- Spearman ---
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(sub)), sub["spearman_norm"])
        plt.xticks(range(len(sub)), pair_labels, rotation=45, ha="right")
        plt.ylim(0.0, 1.05)
        plt.ylabel("Spearman (normalized)")
        plt.title(f"Spearman distances for ID {id_}")
        plt.tight_layout()

        fname_s = f"per_id_spearman_{id_}.png"
        out_path_s = os.path.join(output_dir, fname_s)
        plt.savefig(out_path_s, dpi=200)
        plt.close()
        print(f"Saved per-ID Spearman plot to {out_path_s}")

        # --- Combined ---
        if "combined_norm" in sub.columns:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(sub)), sub["combined_norm"])
            plt.xticks(range(len(sub)), pair_labels, rotation=45, ha="right")
            plt.ylim(0.0, 1.05)
            plt.ylabel("Combined (normalized)")
            plt.title(f"Combined distances for ID {id_}")
            plt.tight_layout()

            fname_c = f"per_id_combined_{id_}.png"
            out_path_c = os.path.join(output_dir, fname_c)
            plt.savefig(out_path_c, dpi=200)
            plt.close()
            print(f"Saved per-ID Combined plot to {out_path_c}")


# -------------------- SUMMARY PER METHOD (COMBINED) --------------------


def generate_combined_summary(df: pd.DataFrame, output_dir: str):
    """
    Create a summary bar plot for the combined metric:
    for each method, compute the average combined_norm
    over all pairs where that method appears (as method_a or method_b).
    """
    if "combined_norm" not in df.columns:
        print("combined_norm column not found in DataFrame. Skipping combined summary.")
        return

    methods = get_all_methods(df)
    avg_values = []

    for m in methods:
        cond = (df["method_a"] == m) | (df["method_b"] == m)
        sub = df[cond]
        if len(sub) == 0:
            continue
        avg_values.append((m, sub["combined_norm"].mean()))

    if not avg_values:
        print("No data for combined summary.")
        return

    methods_list = [m for m, _ in avg_values]
    values_list = [v for _, v in avg_values]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(methods_list)), values_list)
    plt.xticks(range(len(methods_list)), methods_list, rotation=45, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Average combined distance (normalized)")
    plt.title("Average combined distance per method (vs all others)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "combined_summary_per_method.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved combined summary plot to {out_path}")


# -------------------- MAIN --------------------


if __name__ == "__main__":
    ensure_output_dir(OUTPUT_DIR)

    df = load_results(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    generate_heatmaps(df, OUTPUT_DIR)
    generate_scatter_plots(df, OUTPUT_DIR)
    generate_per_id_plots(df, OUTPUT_DIR)
    generate_combined_summary(df, OUTPUT_DIR)
