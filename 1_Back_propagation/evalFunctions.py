import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    acc = np.mean(LPred == LTrue)
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """
    labels = np.unique(np.concatenate((LPred, LTrue)))
    n_labels = len(labels)
    cM = np.zeros((n_labels, n_labels), dtype=int)
    for i, pred_label in enumerate(labels):
        for j, true_label in enumerate(labels):
            cM[i, j] = np.sum((LPred == pred_label) & (LTrue == true_label))

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    acc = np.trace(cM) / np.sum(cM)
    return acc
