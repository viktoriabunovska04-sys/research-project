"""Compute and compare standard metrics across definition conditions for Flan-T5/Qwen.

This script loads multiple result CSV files depending on the chosen model, computes standard
classification metrics for each dataset, and compares the model's
predictions with the baseline no-definition predictions.

It produces a summary table of accuracy, precision, recall, false positive
rate, and F1 score, then visualizes confusion matrices for both the baseline
and given-definition predictions.
"""

import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.eval_metrics_utils import compute_accuracy, compute_recall, compute_precision, compute_false_positives, compute_f1
from utils.general_utils import load_results

# choose which model you want to see flip analysis for
MODEL = "flan"
# MODEL = "qwen"

results = load_results(f"results/predictions_{MODEL}", MODEL)

datasets = [
    ("Bulgaria", results["Bulgaria"]),
    ("Croatia", results["Croatia"]),
    ("Meta", results["Meta"]),
    ("Reddit", results["Reddit"]),
    ("Theoretical Inclusion", results["Theoretic I"]),
    ("Theoretical Inclusion + Exclusion", results["Theoretic I + E"]),
]

dataset_own_results = results["No definition"]

rows = []

for name, dataset in datasets:
    # Own-definition metrics
    own_accuracy = compute_accuracy(
        dataset_own_results["prediction"],
        dataset["true_label"]
    )

    own_precision = compute_precision(
        dataset_own_results["prediction"],
        dataset["true_label"]
    )

    own_recall = compute_recall(
        dataset_own_results["prediction"],
        dataset["true_label"]
    )

    own_fpr = compute_false_positives(
        dataset_own_results["prediction"],
        dataset["true_label"]
    )

    own_f1 = compute_f1(own_precision, own_recall)

    # Given-definition metrics
    given_accuracy = compute_accuracy(
        dataset["prediction"],
        dataset["true_label"]
    )

    given_precision = compute_precision(
        dataset["prediction"],
        dataset["true_label"]
    )

    given_recall = compute_recall(
        dataset["prediction"],
        dataset["true_label"]
    )

    given_fpr = compute_false_positives(
        dataset["prediction"],
        dataset["true_label"]
    )

    given_f1 = compute_f1(given_precision, given_recall)

    rows.append({
        "Dataset": name,

        "Own Accuracy": own_accuracy,
        "Given Accuracy": given_accuracy,
        "ΔAccuracy": given_accuracy - own_accuracy,

        "Own Precision": own_precision,
        "Given Precision": given_precision,
        "ΔPrecision": given_precision - own_precision,

        "Own Recall": own_recall,
        "Given Recall": given_recall,
        "ΔRecall": given_recall - own_recall,

        "Own FPR": own_fpr,
        "Given FPR": given_fpr,
        "ΔFPR": given_fpr - own_fpr,

        "Own F1": own_f1,
        "Given F1": given_f1,
        "ΔF1": given_f1 - own_f1
    })

results_table = pd.DataFrame(rows).round(4)
results_table.to_html("results/tables/results_table.html")


#--------Confusion Matrices-------#

# def rate_matrix(cm):
#     """
#     Normalize a confusion matrix row-wise into per-class rates.
#     """
#     cm = cm.astype(float)
#     return cm / cm.sum(axis=1, keepdims=True)

# fig, axes = plt.subplots(len(datasets), 4, figsize=(16, 4 * len(datasets)))

# for i, (name, dataset) in enumerate(datasets):

#     # Confusion matrices
#     cm_own = confusion_matrix(
#         dataset["true_label"],
#         dataset_own_results["prediction"]
#     )

#     cm_given = confusion_matrix(
#         dataset["true_label"],
#         dataset["prediction"]
#     )

#     # Rate matrices
#     rm_own = rate_matrix(cm_own)
#     rm_given = rate_matrix(cm_given)

#     ConfusionMatrixDisplay(
#         cm_own,
#         display_labels=[0, 1]
#     ).plot(ax=axes[i, 0], colorbar=False)
#     axes[i, 0].set_title(f"{name} – Own")

#     ConfusionMatrixDisplay(
#         cm_given,
#         display_labels=[0, 1]
#     ).plot(ax=axes[i, 1], colorbar=False)
#     axes[i, 1].set_title(f"{name} – Given")

#     ConfusionMatrixDisplay(
#         rm_own,
#         display_labels=[0, 1]
#     ).plot(ax=axes[i, 2], values_format=".2f", colorbar=False)
#     axes[i, 2].set_title(f"{name} – Own (Rates)")

#     ConfusionMatrixDisplay(
#         rm_given,
#         display_labels=[0, 1]
#     ).plot(ax=axes[i, 3], values_format=".2f", colorbar=False)
#     axes[i, 3].set_title(f"{name} – Given (Rates)")

# plt.tight_layout()
# plt.show()