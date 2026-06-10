"""Compare FLAN and Qwen flips for qualitative sample analysis.

This module loads saved FLAN and Qwen prediction results, identifies gender
samples with beneficial definition-induced flips, and assembles a small sample
of cases for qualitative comparison.
"""

from utils.general_utils import load_results
import pandas as pd

results_flan = load_results(f"results/predictions_flan", "flan")
results_qwen = load_results(f"results/predictions_qwen", "qwen")

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


def get_flip_indices(datasets, dataset_own, n):
    """Select a sample of gender-related indices with beneficial flips.

    This function checks that all definition-conditioned datasets share the same
    test cases, filters to gender-targeted examples, and returns a random
    sample of indices for which at least one definition-conditioned model
    corrected a baseline prediction.

    Args:
        datasets (list[pd.DataFrame]): Definition-conditioned result datasets.
        dataset_own (pd.DataFrame): Baseline no-definition dataset.
        n (int): Desired number of sample indices.

    Returns:
        np.ndarray: Randomly sampled row indices for qualitative review.
    """
    dataset = datasets[0]
    
    for d in datasets:
        if any(d["test_case"] != dataset["test_case"]):
            raise ValueError("Different test cases in datasets")
    
    dataset_gender = dataset[dataset["target_type"] == "gender"]
    gender_indices = dataset_gender.index
    
    beneficial_mask = pd.Series(False, index=gender_indices)
    for d in datasets:
        flip = (d.loc[gender_indices, "prediction"] != dataset_own.loc[gender_indices, "prediction"])
        beneficial = flip & (d.loc[gender_indices, "prediction"] == d.loc[gender_indices, "true_label"])
        beneficial_mask = beneficial_mask | beneficial
    
    flip_indices = beneficial_mask[beneficial_mask].index
    print(f"Found {len(flip_indices)} gender samples with at least one beneficial flip")
    
    sample_indices = pd.Series(flip_indices).sample(min(n, len(flip_indices)), random_state=10).values
    return sample_indices


def pick_random_gender_flips(datasets, names, sample_indices, model, dataset_own):
    """Build a side-by-side comparison table for sampled flip cases.

    The returned DataFrame includes the chosen sample text and the predictions
    and true labels across each definition-conditioned run, plus the original
    no-definition baseline prediction.

    Args:
        datasets (list[pd.DataFrame]): Definition-conditioned result datasets.
        names (list[str]): Human-readable names for the datasets.
        sample_indices (Sequence[int]): Preselected row indices.
        model (str): Model identifier, used for naming contextual columns.
        dataset_own (pd.DataFrame): Baseline no-definition dataset.

    Returns:
        pd.DataFrame: A comparison table for selected flip examples.
    """
    dataset = datasets[0]
    
    result = dataset.loc[sample_indices][["functionality", "test_case"]].copy()
    result[f"{names[0]}_pred"] = dataset.loc[sample_indices]["prediction"]
    result[f"{names[0]}_true"] = dataset.loc[sample_indices]["true_label"]
    
    for d, name in zip(datasets[1:], names[1:]):
        result[f"{name}_pred"] = d.loc[sample_indices]["prediction"]
        result[f"{name}_true"] = d.loc[sample_indices]["true_label"]

    result[f"own_prediction"] = dataset_own.loc[sample_indices]["prediction"]
    return result


sample_indices = get_flip_indices(datasets_flan, dataset_own_results_flan, 10)

flan_samples = pick_random_gender_flips(datasets_flan, names, sample_indices, "flan", dataset_own_results_flan)
qwen_samples = pick_random_gender_flips(datasets_qwen, names, sample_indices, "qwen", dataset_own_results_qwen)

flan_samples.to_html("results/tables/flan_samples.html")
qwen_samples.to_html("results/tables/qwen_samples.html")
