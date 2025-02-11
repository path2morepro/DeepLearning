import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = np.mean(LPred == LTrue)
    # ============================================
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

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    TP = np.sum((LTrue == 1) & (LPred == 1))
    TN = np.sum((LTrue == 0) & (LPred == 0))
    FP = np.sum((LTrue == 0) & (LPred == 1))
    FN = np.sum((LTrue == 1) & (LPred == 0))
    cM = np.array([[TP, FN],
                    [FP, TN]])
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # ------------1--------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = cM[0][0] / sum(cM)
    # ============================================
    
    return acc
