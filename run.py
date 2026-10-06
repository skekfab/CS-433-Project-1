import numpy as np
import matplotlib.pyplot as plt

from helpers import create_csv_submission, build_k_indices
import implementations as impl
import data_construction as Data
import model as Model


def main():
    #
    # 1. Load and prepare data
    #
    tx_train, tx_test, y_train, test_ids = Data.prepare_data(*Data.load_csv_data_cached())


    #
    # 2. Train with cross validation
    #
    # TODO currentlyy not used as their is now hyperparameter
    SEED = 42
    K_FOLD = 2
    MAX_ITERS = 1000
    STEP_SIZE = 0.4
    INITIAL_W = np.array([-5.0, 4.0])

    # k_indices = build_k_indices(y_train, K_FOLD, SEED)
    #
    # for k in range(0, K_FOLD):
    #     #
    #     # Perform one whole training
    #     #
    #     tr_indices = k_indices[k]
    #     val_indices = np.concatenate([k_indices[j] for j in range(len(k_indices)) if j != k])
    #
    #     # Split training data into tr and val
    #     tx_tr = tx_train[tr_indices]
    #     y_tr = y_train[tr_indices]
    #     tx_val = tx_train[val_indices]
    #     y_val = tx_train[val_indices]
    #
    #     # Fit model
    #     w, logistic_loss = impl.logistic_regression(y_tr, tx_tr, INITIAL_W, MAX_ITERS, STEP_SIZE)

    #
    # 3. Select best hyperparameter based on loss and retrain model on whole train dataset
    #
    # TODO for now we do not have a hyperparameter -> just retrain
    w, log_loss = impl.logistic_regression(y_train, tx_train, INITIAL_W, MAX_ITERS, STEP_SIZE)

    #
    # 4. Find decision threshold
    #
    precs = []
    recs = []
    f1s = []
    accs = []

    THRESHOLDS = np.linspace(0, 1, 100)

    for threshold_index, threshold in enumerate(THRESHOLDS):

        # Use decision function with threshold to get predictions
        y_tr_pred = Model.predict_logistic(w, tx_train, threshold)

        # Calculate confusion matrix
        tp, fp, fn, tn = Model.confusion_matrix(y_tr_pred, y_train)

        # Calculate precision, recall, f1-score, and recall
        acc, f1, prec, rec = Model.evaluate_model(fn, fp, tn, tp)

        # Append precision, recall, f1-score, and recall to overall list
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        accs.append(acc)

    # Find best threshold
    best_threshold_index = np.argmax(f1s)
    best_threshold = THRESHOLDS[best_threshold_index]
    print(f"Best threshold: {best_threshold:.2f}")
    print("Results:")
    print(f"  Precision {precs[best_threshold_index]:.2f}")
    print(f"  Recall {precs[best_threshold_index]:.2f}")
    print(f"  F1-Score {precs[best_threshold_index]:.2f}")
    print(f"  Accuracy {precs[best_threshold_index]:.2f}")

    #
    # 5. Run model on test data and produce submission txt
    #
    y_pred = Model.predict_logistic(w, tx_test, best_threshold)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print("submission.csv created !")


    #
    # 6. Create plot with decision boundary
    #
    age_range = np.arange(0, 81)
    prob_curve = impl.sigmoid(w[0] + w[1] * (age_range / 80))

    decision_age = (np.log(best_threshold / (1 - best_threshold)) - w[0]) / w[1] * 80

    ages = tx_train[:, 1] * 80
    labels_01 = (y_train + 1) / 2  # map {-1, 1} -> {0, 1} for plotting
    plt.scatter(ages[y_train == -1], labels_01[y_train == -1], alpha=0.05, s=5, color="steelblue", label="No disease")
    plt.scatter(ages[y_train ==  1], labels_01[y_train ==  1], alpha=0.05, s=5, color="salmon",    label="Disease")
    plt.plot(age_range, prob_curve, color="black", linewidth=2, label="P(disease | age)")
    plt.axvline(decision_age, color="red", linestyle="--", label=f"Decision boundary (age={decision_age:.1f})")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Age")
    plt.ylabel("P(heart disease = 1)")
    plt.legend()
    plt.savefig("decision_boundary.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
