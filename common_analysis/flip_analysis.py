"""Generate prediction comparison plots and flip statistics for Qwen/FLAN results.

This script loads saved model results for different definition conditions, plots
binary prediction distributions across definitions, and computes flip statistics
against a baseline "No definition" prediction set.
"""

import pandas as pd
from utils.eval_flips_utils import compute_flip_stats
from utils.general_utils import load_results

# choose which model you want to see flip analysis for
MODEL="flan"
# MODEL="qwen"

datasets = load_results(f"results/predictions_{MODEL}", MODEL)

dataset_bulgaria_results = datasets["Bulgaria"]
dataset_croatia_results = datasets["Croatia"]
dataset_meta_results = datasets["Meta"]
dataset_reddit_results = datasets["Reddit"]
dataset_theoretical_incl_results = datasets["Theoretic I"]
dataset_theoretical_incl_excl_results = datasets["Theoretic I + E"]
dataset_own_results = datasets["No definition"]

datasets = [dataset_bulgaria_results, dataset_croatia_results, dataset_meta_results, 
           dataset_reddit_results, dataset_theoretical_incl_results, dataset_theoretical_incl_excl_results, dataset_own_results]
names = ["Bulgaria", "Croatia", "Meta", "Reddit", "Theoretic I", "Theoretic I+E", "No definition"]


#flip-rate
flip_bulgaria = compute_flip_stats(
    dataset_own_results["prediction"],
    dataset_bulgaria_results["prediction"],
    dataset_bulgaria_results["true_label"]
)

flip_croatia = compute_flip_stats(
    dataset_own_results["prediction"],
    dataset_croatia_results["prediction"],
    dataset_croatia_results["true_label"]
)

flip_meta = compute_flip_stats(
    dataset_own_results["prediction"],
    dataset_meta_results["prediction"],
    dataset_meta_results["true_label"]  
)

flip_reddit = compute_flip_stats(
    dataset_own_results["prediction"],
    dataset_reddit_results["prediction"],
    dataset_reddit_results["true_label"]  
)

flip_theoretical_incl = compute_flip_stats(
    dataset_own_results["prediction"],
    dataset_theoretical_incl_results["prediction"],
    dataset_theoretical_incl_results["true_label"]  
)

flip_theoretical_incl_excl = compute_flip_stats(
    dataset_own_results["prediction"],
    dataset_theoretical_incl_excl_results["prediction"],
    dataset_theoretical_incl_excl_results["true_label"]  
)

flip_table = pd.DataFrame({
    "Dataset": [
        "Bulgaria",
        "Croatia",
        "Meta",
        "Reddit",
        "Theoretical Inclusion",
        "Theoretical Inclusion + Exclusion"
    ],
    "Flip Rate": [
        flip_bulgaria["flip_rate"],
        flip_croatia["flip_rate"],
        flip_meta["flip_rate"],
        flip_reddit["flip_rate"],
        flip_theoretical_incl["flip_rate"],
        flip_theoretical_incl_excl["flip_rate"]
    ],
    "Num Flips": [
        flip_bulgaria["num_flips"],
        flip_croatia["num_flips"],
        flip_meta["num_flips"],
        flip_reddit["num_flips"],
        flip_theoretical_incl["num_flips"],
        flip_theoretical_incl_excl["num_flips"]
    ],
    "Beneficial Flips": [
        flip_bulgaria["num_beneficial_flips"],
        flip_croatia["num_beneficial_flips"],
        flip_meta["num_beneficial_flips"],
        flip_reddit["num_beneficial_flips"],
        flip_theoretical_incl["num_beneficial_flips"],
        flip_theoretical_incl_excl["num_beneficial_flips"]
    ],
    "Beneficial Flip Rate": [
        flip_bulgaria["beneficial_flip_rate"],
        flip_croatia["beneficial_flip_rate"],
        flip_meta["beneficial_flip_rate"],
        flip_reddit["beneficial_flip_rate"],
        flip_theoretical_incl["beneficial_flip_rate"],
        flip_theoretical_incl_excl["beneficial_flip_rate"]
    ],
    "Harmful Flips": [
        flip_bulgaria["num_harmful_flips"],
        flip_croatia["num_harmful_flips"],
        flip_meta["num_harmful_flips"],
        flip_reddit["num_harmful_flips"],
        flip_theoretical_incl["num_harmful_flips"],
        flip_theoretical_incl_excl["num_harmful_flips"]
    ],
    "Harmful Flip Rate": [
        flip_bulgaria["harmful_flip_rate"],
        flip_croatia["harmful_flip_rate"],
        flip_meta["harmful_flip_rate"],
        flip_reddit["harmful_flip_rate"],
        flip_theoretical_incl["harmful_flip_rate"],
        flip_theoretical_incl_excl["harmful_flip_rate"]
    ],
    "Total Samples": [
        flip_bulgaria["total"],
        flip_croatia["total"],
        flip_meta["total"],
        flip_reddit["total"],
        flip_theoretical_incl["total"],
        flip_theoretical_incl_excl["total"]
    ]
})

flip_table["Flip Rate"] = flip_table["Flip Rate"].round(4)
flip_table.to_html("results/tables/flip_table.html")
