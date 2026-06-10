"""Relabel the extended HateCheck dataset constaining dominant samples
    under alternative definitions.

This script reads the HateCheck dataset, applies dataset-specific
rules to compute a new `def_label` column under several definitions
(Bulgaria, Croatia, Meta, Reddit, and Theoretic), and writes the relabeled
frames back to separate Excel files.
"""

import pandas as pd
import numpy as np

path = "data/hatecheck_with_dominant.xlsx"
dataset = pd.read_excel(path, engine="openpyxl")


def values(dataset):
    """Extract normalized indicator columns from the dataset.

    This helper strips whitespace from the relevant annotation fields so the
    relabeling heuristics can consistently apply regex matching.

    Args:
        dataset (pd.DataFrame): The extended HateCheck dataset.

    Returns:
        tuple: Normalized series for target_type, dominance, explicit_ref,
            incites, group_insult, and in_group.
    """
    target_type = dataset["target_type"].str.strip()
    dominance = dataset["dominance"].str.strip()
    reference = dataset["explicit_ref"].str.strip()
    consequences = dataset["incites"].str.strip()
    group_insult = dataset["group_insult"].str.strip()
    in_group = dataset["in_group"].str.strip()
    return  target_type, dominance, reference, consequences, group_insult, in_group


def relabel_bulgaria(dataset):
    """Apply the Bulgaria definition decomposition and save the relabeled file.

    Args:
        dataset (pd.DataFrame): Extended HateCheck dataset.

    Returns:
        pd.DataFrame: Copy of the dataset with a new `def_label` assignment.
    """
    copy = dataset.copy(deep=True)
    target_type, dominance, reference, consequences, group_insult, in_group = values(copy)

    target_type_filter = target_type.str.contains("race|religion|nationality", case=False, na=False)
    reference_filter = reference.str.contains("group_characteristic|slur|stereotype", case=False, na=False)
    consequences_filter = consequences.str.contains("violence|discrimination|hate", case=False, na=False)
    group_insult_filter = group_insult.str.contains("yes", case=False, na=False)
    
    copy["def_label"] = np.where(target_type_filter & reference_filter & (consequences_filter | group_insult_filter), "hateful", "non-hateful")
    copy.to_excel("data/hatecheck_with_dominant_bulgaria.xlsx", sheet_name="Bulgaria", index=False)
    return copy


def relabel_croatia(dataset):
    """Apply the Croatia definition decomposition and save the relabeled file."""
    copy = dataset.copy(deep=True)
    target_type, dominance, reference, consequences, group_insult, in_group = values(copy)

    reference_filter = reference.str.contains("group_characteristic|slur|stereotype", case=False, na=False)
    consequences_filter = consequences.str.contains("violence|hate", case=False, na=False)
    group_insult_filter = group_insult.str.contains("yes", case=False, na=False)
    
    copy["def_label"] = np.where(reference_filter & (consequences_filter | group_insult_filter), "hateful", "non-hateful")
    copy.to_excel("data/hatecheck_with_dominant_croatia.xlsx", sheet_name="Croatia", index=False)
    return copy


def relabel_meta(dataset):
    """Apply the Meta definition decomposition and save the relabeled file."""
    copy = dataset.copy(deep=True)
    target_type, dominance, reference, consequences, group_insult, in_group = values(copy)

    target_type_filter = target_type.str.contains("gender|sexual orientation|race|disability|religion|nationality", case=False, na=False)
    reference_filter = reference.str.contains("group_characteristic|slur|stereotype", case=False, na=False)
    consequences_filter = consequences.str.contains("violence|discrimination|hate", case=False, na=False)
    group_insult_filter = group_insult.str.contains("yes", case=False, na=False)
    in_group_filter = in_group.str.contains("yes", case=False, na=False)
    
    copy["def_label"] = np.where(in_group_filter, "non-hateful", np.where(target_type_filter & reference_filter & (consequences_filter | group_insult_filter), "hateful", "non-hateful"))
    copy.to_excel("data/hatecheck_with_dominant_meta.xlsx", sheet_name="Meta", index=False)
    return copy


def relabel_reddit(dataset):
    """Apply the Reddit definition decomposition and save the relabeled file."""
    copy = dataset.copy(deep=True)
    target_type, dominance, reference, consequences, group_insult, in_group = values(copy)

    dominance_filter = dominance.str.contains("no", case=False, na=False)
    reference_filter = reference.str.contains("group_characteristic|slur|stereotype", case=False, na=False)
    consequences_filter = consequences.str.contains("violence|hate", case=False, na=False)
    group_insult_filter = group_insult.str.contains("yes", case=False, na=False)
    
    copy["def_label"] = np.where(dominance_filter & reference_filter & (consequences_filter | group_insult_filter), "hateful", "non-hateful")
    copy.to_excel("data/hatecheck_with_dominant_reddit.xlsx", sheet_name="Reddit", index=False)
    return copy


def relabel_theoretical(dataset):
    """Apply the theoretical definition decomposition and save the relabeled file."""
    copy = dataset.copy(deep=True)
    target_type, dominance, reference, consequences, group_insult, in_group = values(copy)

    target_type_filter = target_type.str.contains("sexual orientation|race|disability|religion|nationality", case=False, na=False)
    dominance_filter = dominance.str.contains("no|yes", case=False, na=False)
    reference_filter = reference.str.contains("group_characteristic|slur", case=False, na=False)
    consequences_filter = consequences.str.contains("violence|hate", case=False, na=False)
    group_insult_filter = group_insult.str.contains("yes", case=False, na=False)
    in_group_filter = in_group.str.contains("yes", case=False, na=False)

    copy["def_label"] = np.where(in_group_filter, "non-hateful", np.where(target_type_filter & dominance_filter & reference_filter & (consequences_filter | group_insult_filter), "hateful", "non-hateful"))
    copy.to_excel("data/hatecheck_with_dominant_theoretical.xlsx", sheet_name="Theoretic", index=False)
    return copy


relabel_bulgaria(dataset)
relabel_croatia(dataset)
relabel_meta(dataset)
relabel_reddit(dataset)
relabel_theoretical(dataset)