import numpy as np

def compute_flip_stats(pred_own, pred_given, true_labels):
    """Compute flip statistics comparing two prediction sets.

    A flip is counted when the `pred_given` result differs from the
    `pred_own` baseline prediction. This function also measures whether
    each flip is beneficial or harmful with respect to the ground truth.

    Args:
        pred_own (Sequence[int]): Baseline predictions without a definition.
        pred_given (Sequence[int]): Predictions produced under a given
            definition condition.
        true_labels (Sequence[int]): Ground truth labels for the same samples.
    """
    if len(pred_own) != len(pred_given):
        raise ValueError("Mismatched prediction lengths")

    pred_own = np.array(pred_own)
    pred_given = np.array(pred_given)
    true_labels = np.array(true_labels)

    flips = np.sum(pred_own != pred_given)
    beneficial_flips = np.sum((pred_own != pred_given) & (pred_given == true_labels))
    harmful_flips = np.sum((pred_own != pred_given) & (pred_given != true_labels))
    total = len(pred_own)

    return {
        "flip_rate": flips / total if flips > 0 else 0.0,
        "num_flips": flips,
        "num_beneficial_flips": beneficial_flips,
        "beneficial_flip_rate": beneficial_flips / flips if flips > 0 else 0.0,
        "num_harmful_flips": harmful_flips,
        "harmful_flip_rate": harmful_flips / flips if flips > 0 else 0.0,
        "total": total
    }