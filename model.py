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


def best_threshold(tx_train, y_train, w):
    
    precs , recs, f1s, accs = [], [], [], []

    THRESHOLDS = np.linspace(0, 1, 100)

    for threshold_index, threshold in enumerate(THRESHOLDS):

        # Use decision function with threshold to get predictions
        y_tr_pred = predict_logistic(w, tx_train, threshold)

        # Calculate confusion matrix
        tp, fp, fn, tn = confusion_matrix(y_tr_pred, y_train)

        # Calculate precision, recall, f1-score, and recall
        acc, f1, prec, rec = evaluate_model(fn, fp, tn, tp)

        # Append precision, recall, f1-score, and recall to overall list
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        accs.append(acc)

    best_threshold_index = np.argmax(f1s)
    best_threshold = THRESHOLDS[best_threshold_index]
    return best_threshold , precs[best_threshold_index], recs[best_threshold_index], f1s[best_threshold_index], accs[best_threshold_index]