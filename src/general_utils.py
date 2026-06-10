import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir, model):
    return {
        "Bulgaria": pd.read_csv(f"{results_dir}/bulgaria_results_{model}.csv"),
        "Croatia": pd.read_csv(f"{results_dir}/croatia_results_{model}.csv"),
        "Meta": pd.read_csv(f"{results_dir}/meta_results_{model}.csv"),
        "Reddit": pd.read_csv(f"{results_dir}/reddit_results_{model}.csv"),
        "Theoretic I": pd.read_csv(f"{results_dir}/theoretical_incl_results_{model}.csv"),
        "Theoretic I + E": pd.read_csv(f"{results_dir}/theoretical_incl_excl_results_{model}.csv"),
        "No definition": pd.read_csv(f"{results_dir}/own_csv_{model}.csv"),
    }

# not used in the paper but gives a nice overview of the diffrences between the two models
def diff_predictions(datasets, names):
    """Plot stacked counts of harmful and non-harmful predictions for each dataset.

    The visualization compares how prediction distributions change across
    definition conditions, making it easier to inspect whether some prompt
    variants produce more harmful or non-harmful outputs.

    Args:
        datasets (list[pd.DataFrame]): Result tables containing a "prediction"
            column with binary labels.
        names (list[str]): Human-readable labels for each dataset.

    Raises:
        ValueError: If the number of dataset frames does not match the number of
            provided labels.
    """
    if len(datasets) != len(names):
        raise ValueError("Mismatched datasets and names lengths")
    harmful = []
    non_harmful = []
    for dataset in datasets:
        harmful.append(np.sum(dataset["prediction"] == 1))
        non_harmful.append(np.sum(dataset["prediction"] == 0))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, harmful)
    ax.bar(names, non_harmful, bottom=harmful)
    ax.set_xlabel("Definitions", fontsize=16)
    ax.set_ylabel("Number of Predictions", fontsize=16)
    ax.legend(["Harmful", "Non-harmful"])
    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=14)

    total = [h + n for h, n in zip(harmful, non_harmful)]
    for i, (h, n, t) in enumerate(zip(harmful, non_harmful, total)):
        ax.text(i, h / 2, f"{h/t*100:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(i, h + n / 2, f"{n/t*100:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    plt.tight_layout()
    plt.show()