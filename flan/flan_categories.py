"""Evaluate FLAN-T5 prediction performance and flip statistics by category.

This script loads FLAN-T5 result CSV files and compares model behavior across
multiple definition conditions using categorical splits such as target
group, dominance, explicit reference, and consequences.
"""

import pandas as pd
from src.eval_metrics_utils import compute_recall, compute_precision, compute_f1
from src.eval_flips_utils import compute_flip_stats
from IPython.display import display
from src.general_utils import load_results
import webbrowser

datasets = load_results(f"results/predictions_flan", "flan")
dataset_own_results = datasets["No definition"]

def split_dataset(dataset, split_category, concrete_split):
    """Filter a dataset to rows where a category contains a split string.

    Args:
        dataset (pandas.DataFrame): The dataset to filter.
        split_category (str): The name of the column to search.
        concrete_split (str): The substring value to match.

    Returns:
        pandas.DataFrame: Filtered rows where the chosen column contains the split.
    """
    return dataset[
        dataset[split_category]
        .str.contains(concrete_split, na=False)
    ]


def compute_metrics(predictions, labels):
    """Compute F1 score from binary predictions and labels.
    """
    precision = compute_precision(predictions, labels)
    recall = compute_recall(predictions, labels)
    return compute_f1(precision, recall)


def build_result(definition_name, group_key, group_value, f1_given, f1_own, flip_stats):
    """Build a single result record for a category evaluation.

    Args:
        definition_name (str): The condition label for the evaluation.
        group_key (str): The key used to name the split dimension.
        group_value (str): The value of the selected split.
        f1_given (float): F1 score for given-definition predictions.
        f1_own (float): F1 score for baseline predictions.
        flip_stats (dict): Flip metric dictionary from compute_flip_stats.

    Returns:
        dict: Structured evaluation metrics and flip statistics.
    """
    f1_delta = f1_given - f1_own
    valid = f1_given > 0

    return {
        "definition_name": definition_name,
        group_key: group_value,
        "f1_given": f1_given  if valid else "--",
        "f1_own": f1_own if valid else "--",
        "ΔF1 (given-own)": f1_delta  if valid else "--",
        "flip_rate": flip_stats["flip_rate"],
        "num_flips": flip_stats["num_flips"],
        "num_beneficial_flips": flip_stats["num_beneficial_flips"],
        "beneficial_flip_rate": flip_stats["beneficial_flip_rate"],
        "num_harmful_flips": flip_stats["num_harmful_flips"],
        "harmful_flip_rate": flip_stats["harmful_flip_rate"],
        "total": flip_stats["total"],
    }


def compute_f1_and_flips(split_own, split_given):
    """Compute F1 scores and flip stats for a pair of splits."""
    f1_given = compute_metrics(split_given["prediction"], split_given["true_label"])
    f1_own = compute_metrics(split_own["prediction"], split_given["true_label"])
    flips = compute_flip_stats(split_own["prediction"], split_given["prediction"], split_given["true_label"])
    return f1_given, f1_own, flips


#-----Evaluate Components------

def evaluate_target_group(dataset_own, dataset_given, target_group, definition_name):
    """Evaluate F1 and flips on samples mentioning a target group.

    This function filters both the baseline and given-definition datasets to
    samples whose `target_type` contains the requested target group. It then
    computes the relevant F1 and flip metrics for that subgroup.

    Args:
        dataset_own (pandas.DataFrame): Baseline dataset without definition.
        dataset_given (pandas.DataFrame): Given-definition dataset.
        target_group (str): The target group label to filter on.
        definition_name (str): Human-readable condition name for reporting.
    """
    dataset_given["target_type"] = dataset_given["target_type"].str.strip()
    dataset_own["target_type"] = dataset_own["target_type"].str.strip()

    split_given = split_dataset(dataset_given, "target_type", target_group)
    split_own = split_dataset(dataset_own, "target_type", target_group)

    f1_given, f1_own, flips = compute_f1_and_flips(split_own, split_given)
    return build_result(definition_name, "target_group", target_group, f1_given, f1_own, flips)


def evaluate_dominance_general(dataset_own, dataset_given, definition_name):
    """Evaluate F1 and flips for samples with a dominance label."""
    # Skip samples with no dominance label (non-hateful, not directed at a target group)
    split_given = dataset_given[dataset_given["dominance"].notna()]
    split_own = dataset_own[dataset_own["dominance"].notna()]
    f1_given, f1_own, flips = compute_f1_and_flips(split_own, split_given)
    return build_result(definition_name, "category", "dominance", f1_given, f1_own, flips)


def evaluate_explicit_ref(dataset_own, dataset_given, reference_type, definition_name):
    """Evaluate F1 and flips for explicit reference categories."""
    split_given = split_dataset(dataset_given, "explicit_reference", reference_type)
    split_own = split_dataset(dataset_own, "explicit_reference", reference_type)

    f1_given, f1_own, flips = compute_f1_and_flips(split_own, split_given)
    return build_result(definition_name, "explicit_reference", reference_type, f1_given, f1_own, flips)


def evaluate_consequences(dataset_own, dataset_given, consequence_type, definition_name):
    """Evaluate F1 and flips for consequence/incitement categories."""
    split_given = split_dataset(dataset_given, "consequences", consequence_type)
    split_own = split_dataset(dataset_own, "consequences", consequence_type)

    f1_given, f1_own, flips = compute_f1_and_flips(split_own, split_given)
    return build_result(definition_name, "consequences/incites", consequence_type, f1_given, f1_own, flips)


# ----- Run Evaluations----------
def save_and_open(results, filename):
    df = pd.DataFrame(results)
    df.to_html(filename)
    webbrowser.open(filename)

# Gender
gender_results = [
    evaluate_target_group(dataset_own_results, datasets[name], "gender", name)
    for name in ["Bulgaria", "Reddit", "Theoretic I", "Theoretic I + E"]
]

# Dominance
dominance_general_results = [
    evaluate_dominance_general(dataset_own_results, datasets[name], name)
    for name in ["Bulgaria", "Reddit", "Theoretic I", "Theoretic I + E"]
]

# Stereotype
stereotype_results = [
    evaluate_explicit_ref(dataset_own_results, datasets[name], "stereotype", name)
    for name in ["Theoretic I", "Theoretic I + E"]
]

# Discrimination
discrimination_results = [
    evaluate_consequences(dataset_own_results, datasets[name], "discrimination", name)
    for name in ["Croatia", "Theoretic I", "Theoretic I + E"]
]


save_and_open(gender_results, "results/tables/gender_results.html")
save_and_open(dominance_general_results, "results/tables/dominance_general_results.html")
save_and_open(stereotype_results, "results/tables/stereotype_results.html")
save_and_open(discrimination_results, "results/tables/discrimination_results.html")



