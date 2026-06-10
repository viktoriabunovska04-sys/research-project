"""Analyze flip distributions between FLAN and Qwen predictions.

This module loads saved model result tables, identifies beneficial and harmful
prediction flips between a definition-conditioned model output and the baseline
"No definition" output, computes flip rates by category, and creates heatmaps
that compare FLAN and Qwen behavior.
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.general_utils import load_results

results_flan = load_results(f"results/flan", "flan")
results_qwen = load_results(f"results/qwen", "qwen")

dataset_bulgaria_results_flan = results_flan["Bulgaria"]
dataset_croatia_results_flan = results_flan["Croatia"]
dataset_meta_results_flan = results_flan["Meta"]
dataset_reddit_results_flan = results_flan["Reddit"]
dataset_theoretical_incl_results_flan = results_flan["Theoretic I"]
dataset_theoretical_incl_excl_results_flan = results_flan["Theoretic I + E"]
dataset_own_results_flan = results_flan["No definition"]


dataset_bulgaria_results_qwen = results_qwen["Bulgaria"]
dataset_croatia_results_qwen = results_qwen["Croatia"]
dataset_meta_results_qwen = results_qwen["Meta"]
dataset_reddit_results_qwen = results_qwen["Reddit"]
dataset_theoretical_incl_results_qwen = results_qwen["Theoretic I"]
dataset_theoretical_incl_excl_results_qwen = results_qwen["Theoretic I + E"]
dataset_own_results_qwen = results_qwen["No definition"]


datasets_flan = [dataset_bulgaria_results_flan, dataset_croatia_results_flan, dataset_meta_results_flan, 
           dataset_reddit_results_flan, dataset_theoretical_incl_results_flan, dataset_theoretical_incl_excl_results_flan]

datasets_qwen = [dataset_bulgaria_results_qwen, dataset_croatia_results_qwen, dataset_meta_results_qwen, 
           dataset_reddit_results_qwen, dataset_theoretical_incl_results_qwen, dataset_theoretical_incl_excl_results_qwen]

names = ["Bulgaria", "Croatia", "Meta", "Reddit", "Theoretic I", "Theoretic I+E"]



def beneficial_flip_analysis(sample_given, sample_own):
    """Return rows where the given definition flipped incorrectly to correctly.

    A flip is considered beneficial when the prediction under the "given"
    definition differs from the baseline own-definition prediction, and the
    given prediction matches the true label.

    Args:
        sample_given (pd.DataFrame): Results for the definition-conditioned run.
        sample_own (pd.DataFrame): Baseline results for the no-definition run.
    """
    if len(sample_given) != len(sample_own):
        raise ValueError("Mismatched prediction lengths")

    mask = (
        (sample_own["prediction"] != sample_given["prediction"]) &
        (sample_given["prediction"] == sample_given["true_label"])
    )

    beneficial_flips = sample_given.loc[mask, ["test_case", "functionality", "target_type", "consequences", "dominance", "explicit_reference"]].copy()
    return beneficial_flips

def harmful_flip_analysis(sample_given, sample_own):
    """Return rows where the given definition flipped from correct to incorrect.

    A harmful flip occurs when the definition-conditioned prediction differs from
    the own-definition baseline and the given prediction is wrong relative to
    the true label.

    Args:
        sample_given (pd.DataFrame): Results for the definition-conditioned run.
        sample_own (pd.DataFrame): Baseline results for the no-definition run.
    """
    if len(sample_given) != len(sample_own):
        raise ValueError("Mismatched prediction lengths")

    mask = (
        (sample_own["prediction"] != sample_given["prediction"]) &
        (sample_given["prediction"] != sample_given["true_label"])
    )

    harmful_flips = sample_given.loc[mask, ["test_case", "functionality", "target_type", "consequences", "dominance", "explicit_reference"]].copy()
    return harmful_flips

def compute_benficial_rates(dataset_given, dataset_own, component):
    """Compute category-specific beneficial flip rates for a result pair.

    The rate is the fraction of samples in each category whose predictions
    changed from baseline to given and became correct under the given condition.

    Args:
        dataset_given (pd.DataFrame): Definition-conditioned results.
        dataset_own (pd.DataFrame): Baseline no-definition results.
        component (str): Column name used to group flip rates.
    """
    total_counts = dataset_given[component].str.strip().value_counts()

    flips = beneficial_flip_analysis(dataset_given, dataset_own)
    flip_counts = flips[component].str.strip().value_counts()

    rate = (flip_counts / total_counts).dropna().sort_values(ascending=True)
    return rate


def compute_harmful_rates(dataset_given, dataset_own, component):
    """Compute category-specific harmful flip rates for a result pair.

    The rate is the fraction of samples in each category whose predictions
    changed from baseline to given and became incorrect under the given
    condition.

    Args:
        dataset_given (pd.DataFrame): Definition-conditioned results.
        dataset_own (pd.DataFrame): Baseline no-definition results.
        component (str): Column name used to group flip rates.
    """
    total_counts = dataset_given[component].str.strip().value_counts()

    flips = harmful_flip_analysis(dataset_given, dataset_own)
    flip_counts = flips[component].str.strip().value_counts()

    rate = (flip_counts / total_counts).dropna().sort_values(ascending=True)
    return rate


def align_rates(*rates):
    """Align multiple category rate series to a common index.

    When comparing FLAN and Qwen rates, each series may contain different
    category labels. This helper builds a unified category index and fills any
    missing categories with zeros so the series can be compared directly.

    Args:
        *rates (pd.Series): One or more category rate series.
    """
    all_categories = set().union(*(r.index for r in rates))
    aligned = [r.reindex(all_categories, fill_value=0)for r in rates]
    return aligned, all_categories


def beneficial_vs_harmful_rates(list_given_flan, dataset_own_flan, list_given_qwen, dataset_own_qwen, component):
    """Compute both beneficial and harmful flip rate series for FLAN and Qwen.

    This function iterates through corresponding definition-conditioned result
    datasets, computes beneficial and harmful rates for each definition, and
    aligns the category indices for direct FLAN/Qwen comparison.

    Args:
        list_given_flan (list[pd.DataFrame]): FLAN definition-conditioned datasets.
        dataset_own_flan (pd.DataFrame): FLAN baseline no-definition results.
        list_given_qwen (list[pd.DataFrame]): Qwen definition-conditioned datasets.
        dataset_own_qwen (pd.DataFrame): Qwen baseline no-definition results.
        component (str): Column name used for grouping rates.
    """
    rates_beneficial_flan = []
    rates_beneficial_qwen = []

    rates_harmful_flan = []
    rates_harmful_qwen = []
    
    for definition_dataset_flan, definition_dataset_qwen in zip(list_given_flan, list_given_qwen): 
        rate_beneficial_flan = compute_benficial_rates(definition_dataset_flan, dataset_own_flan, component)
        rate_beneficial_qwen = compute_benficial_rates(definition_dataset_qwen, dataset_own_qwen, component)
        (rate_beneficial_flan,rate_beneficial_qwen), categories = align_rates(rate_beneficial_flan,rate_beneficial_qwen)
        rates_beneficial_flan.append(rate_beneficial_flan)
        rates_beneficial_qwen.append(rate_beneficial_qwen)
      
        rate_harmful_flan = compute_harmful_rates(definition_dataset_flan, dataset_own_flan, component)
        rate_harmful_qwen = compute_harmful_rates(definition_dataset_qwen, dataset_own_qwen, component)
        (rate_harmful_flan,rate_harmful_qwen), categories = align_rates(rate_harmful_flan,rate_harmful_qwen)
 
        rates_harmful_flan.append(rate_harmful_flan)
        rates_harmful_qwen.append(rate_harmful_qwen)

    categories_beneficial = rate_beneficial_flan.index
    categories_harmful = rate_harmful_flan.index
    return rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful



def heatmap_rates(rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful, definition_names, component):
    """Plot comparative heatmaps for beneficial and harmful flip rates.

    This function creates a 2x2 grid of heatmaps showing FLAN and Qwen rates for
    beneficial and harmful flips across category labels and definition names.

    Args:
        rates_beneficial_flan (list[pd.Series]): Beneficial flip rates for FLAN.
        rates_beneficial_qwen (list[pd.Series]): Beneficial flip rates for Qwen.
        rates_harmful_flan (list[pd.Series]): Harmful flip rates for FLAN.
        rates_harmful_qwen (list[pd.Series]): Harmful flip rates for Qwen.
        categories_beneficial (Index): Category labels for beneficial flip heatmaps.
        categories_harmful (Index): Category labels for harmful flip heatmaps.
        definition_names (list[str]): Names of the definition conditions.
        component (str): Name of the component used for file naming.
    """
    def to_matrix(rates_list, categories):
        return np.array([r.reindex(categories, fill_value=0).values for r in rates_list])

    mat_ben_flan = to_matrix(rates_beneficial_flan, categories_beneficial)
    mat_ben_qwen = to_matrix(rates_beneficial_qwen, categories_beneficial)
    mat_harm_flan = to_matrix(rates_harmful_flan, categories_harmful)
    mat_harm_qwen = to_matrix(rates_harmful_qwen, categories_harmful)

    vmax_ben = max(mat_ben_flan.max(),  mat_ben_qwen.max())
    vmax_harm = max(mat_harm_flan.max(), mat_harm_qwen.max())
    vmax_global = max(vmax_ben,vmax_harm)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Beneficial (top) vs Harmful (bottom) rates by {component} for Flan (left) and Qwen (right)", fontsize=14)

    panels = [
        (axes[0, 0], mat_ben_flan,  categories_beneficial, vmax_ben, ""),
        (axes[0, 1], mat_ben_qwen,  categories_beneficial, vmax_ben, ""),
        (axes[1, 0], mat_harm_flan, categories_harmful, vmax_harm, ""),
        (axes[1, 1], mat_harm_qwen, categories_harmful, vmax_harm, ""),
    ]

    for ax, matrix, cats, vmax, title in panels:
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=vmax, cmap="Blues")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(len(definition_names)))
        ax.set_yticklabels(definition_names, fontsize=12)

        for i in range(len(definition_names)):
            for j in range(len(cats)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if matrix[i, j] > vmax * 0.6 else "black")


    plt.subplots_adjust(hspace=0.7,wspace=0.3)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=vmax_global))
    fig.colorbar(sm, cax=cbar_ax, label="rate")
    plt.savefig(
        f"results/distributions/beneficial_and_harmful/heatmap_compare_{component}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def heatmap_beneficial_rates(rates_beneficial_flan, rates_beneficial_qwen, categories, definition_names, component):
    """Plot side-by-side heatmaps for beneficial flip rates only.

    This helper visualizes positive model changes by category for FLAN and Qwen
    across the different definition conditions.

    Args:
        rates_beneficial_flan (list[pd.Series]): Beneficial flip rates for FLAN.
        rates_beneficial_qwen (list[pd.Series]): Beneficial flip rates for Qwen.
        categories (Index): Category labels used for both datasets.
        definition_names (list[str]): Names of the definition conditions.
        component (str): Name of the component used for file naming.
    """
    def to_matrix(rates_list):
        return np.array([r.reindex(categories, fill_value=0).values for r in rates_list])
 
    mat_ben_flan  = to_matrix(rates_beneficial_flan)
    mat_ben_qwen  = to_matrix(rates_beneficial_qwen)

    vmax = max(mat_ben_flan.max(), mat_ben_qwen.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Beneficial flip distribution by {component} for Flan (left) and Qwen (right)", fontsize=14)

    panels = [
        (axes[0], mat_ben_flan, ""),
        (axes[1], mat_ben_qwen, ""),
    ]

    for ax, matrix, title in panels:
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=vmax, cmap="Blues")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(len(definition_names)))
        ax.set_yticklabels(definition_names, fontsize=12)

        for i in range(len(definition_names)):
            for j in range(len(categories)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=12, color="white" if matrix[i, j] > vmax * 0.6 else "black")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="rate")
    plt.subplots_adjust(hspace=0.6,wspace=0.4)
    plt.savefig(
        f"results/distributions/beneficial/heatmap_beneficial_{component}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful = beneficial_vs_harmful_rates(datasets_flan, dataset_own_results_flan, datasets_qwen,dataset_own_results_qwen, "target_type")
heatmap_rates(rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful, names, "target_type")
heatmap_beneficial_rates(rates_beneficial_flan, rates_beneficial_qwen, categories_beneficial, names, "target_type")


rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful = beneficial_vs_harmful_rates(datasets_flan, dataset_own_results_flan, datasets_qwen,dataset_own_results_qwen, "dominance")
heatmap_rates(rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful, names, "dominance")
heatmap_beneficial_rates(rates_beneficial_flan, rates_beneficial_qwen, categories_beneficial, names, "dominance")

rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful = beneficial_vs_harmful_rates(datasets_flan, dataset_own_results_flan, datasets_qwen,dataset_own_results_qwen, "consequences")
heatmap_rates(rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful, names, "consequences")
heatmap_beneficial_rates(rates_beneficial_flan, rates_beneficial_qwen, categories_beneficial, names, "consequences")


rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful = beneficial_vs_harmful_rates(datasets_flan, dataset_own_results_flan, datasets_qwen,dataset_own_results_qwen, "explicit_reference")
heatmap_rates(rates_beneficial_flan, rates_beneficial_qwen, rates_harmful_flan, rates_harmful_qwen, categories_beneficial, categories_harmful, names, "explicit_reference")
heatmap_beneficial_rates(rates_beneficial_flan, rates_beneficial_qwen, categories_beneficial, names, "explicit_reference")



