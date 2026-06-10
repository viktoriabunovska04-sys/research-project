"""Utility functions for binary classification metrics.

This module provides manual implementations of common metrics such as
accuracy, recall, precision, false positive rate, and F1 score for hate
speech classification results.
"""

import numpy as np

def compute_accuracy(predictions, true_labels):
    """Compute the fraction of correct predictions."""
    if len(predictions) != len(true_labels): 
        return ValueError("Mismatched lengths")
    correct_pred = 0
    for (p, l) in zip(predictions, true_labels):
        if p == l: 
            correct_pred += 1
    return correct_pred/len(predictions)

# True positive rate (hate speech correctly flagged)
def compute_recall(predictions, true_labels):
    """Compute recall for the positive class (hate speech)."""
    if len(predictions) != len(true_labels): 
        return ValueError("Mismatched lengths")
    correct_pred = 0
    true_positives = 0
    for (p, l) in zip(predictions, true_labels):
        if l == 1: 
            true_positives += 1
            if p == l:
                correct_pred += 1
    if true_positives == 0:
        return 0.0
    else:
        return correct_pred/true_positives

# actual positives / all positives predicted
def compute_precision(predictions, true_labels):
    """Compute precision for the positive class."""
    if len(predictions) != len(true_labels): 
        return ValueError("Mismatched lengths")
    all_positives = 0
    true_positives = 0
    for (p, l) in zip(predictions, true_labels):
        if p == 1:
            all_positives += 1
            if l == 1: 
                true_positives += 1
    if all_positives == 0:
        return 0.0
    else:
        return true_positives/all_positives

def compute_false_positives(predictions, true_labels):
    """Compute the false positive rate for the negative class."""
    if len(predictions) != len(true_labels): 
        return ValueError("Mismatched lengths")
    all_negatives = 0
    false_positives = 0
    for (p, l) in zip(predictions, true_labels):
        if l == 0: 
            all_negatives += 1
            if p == 1:
                false_positives += 1
    if all_negatives == 0:
        return 0.0
    else:
        return false_positives/all_negatives
        
def compute_f1(precision, recall):
    """Compute the harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (2*precision*recall)/ (precision + recall)

