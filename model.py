import numpy as np

from implementations import sigmoid


def predict_logistic(w, tx, threshold):
    y_prop = sigmoid(tx @ w)
    return (y_prop >= threshold).astype(int)


def confusion_matrix(y_prediction, y_true):
    tp = np.sum((y_prediction == 1) & (y_true == 1))
    fp = np.sum((y_prediction == 1) & (y_true == 0))
    fn = np.sum((y_prediction == 0) & (y_true == 1))
    tn = np.sum((y_prediction == 0) & (y_true == 0))
    return tp, fp, fn, tn


def evaluate_model(fn, fp, tn, tp):
    if tp + fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if tp + fn == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)

    if prec + rec == 0:
        f1 = 0
    else:
        f1 = (2 * prec * rec) / (prec + rec)

    acc = (tp + tn) / (tp + fp + fn + tn)

    return acc, f1, prec, rec
