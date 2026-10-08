import numpy as np
import matplotlib.pyplot as plt
from helpers import create_csv_submission, build_k_indices
import implementations as impl
import data_construction as Data
import model as Model

def main():

    # 1. Finding the best hyperparameter lambda for regularized logistic regression using cross-validation and stochastic gradient descent

    tx_train, tx_test, y_train, test_ids = Data.prepare_data_all_non_null_features(*Data.load_csv_data_cached())

    SEED , K_FOLD , MAX_ITERS , GAMMA , MINIBATCH = 42 , 3 , 10000 , 0.05 , 20
    INITIAL_W , LAMBDAS = np.random.uniform(-5, 5, tx_train.shape[1]) , [0, 0.01, 0.02]

    k_indices = build_k_indices(y_train, K_FOLD, SEED)

    loss_averages = []
    for lambda_index, lambda_ in enumerate(LAMBDAS):
        losses = []
        for k in range(0, K_FOLD):
            INITIAL_W = np.random.uniform(-5, 5, tx_train.shape[1])
            tx_tr , y_tr , tx_val , y_val = Data.data_cross_validationed(tx_train, y_train, k_indices, k)
            w, _ = impl.reg_logistic_regression_stochastic(y_tr, tx_tr, lambda_, INITIAL_W, MAX_ITERS, GAMMA, MINIBATCH)
            log_loss_val = impl.logistic_loss(tx_val, y_val, w)
            losses.append(log_loss_val)
        loss_averages.append(np.mean(losses))


    # 2. Select best hyperparameter based on loss and retrain model on whole train dataset and plot training and validation loss over iterations

    best_lambda_index = np.argmin(loss_averages)
    best_lambda = LAMBDAS[best_lambda_index]
    print(f"Best lambda: {best_lambda}")

    train_losses , val_losses = [] , []

    w = np.random.uniform(-5, 5, tx_train.shape[1])

    k_indices = build_k_indices(y_train, 5, SEED)

    tx_tr , y_tr , tx_val , y_val = Data.data_cross_validationed(tx_train, y_train, k_indices, 0)

    tx_tr, y_tr = tx_tr[:300], y_tr[:300]

    for i in range(MAX_ITERS):
        w , loss_tr = impl.reg_logistic_regression_stochastic(y_tr, tx_tr, best_lambda, w, 10, GAMMA, MINIBATCH)
        train_losses.append(loss_tr)
        loss_val = impl.logistic_loss(tx_val, y_val, w)
        val_losses.append(loss_val)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Log-loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.grid(alpha=0.3)
    plt.show()


    # 3. Find decision threshold
    
    x = Model.best_threshold(tx_train, y_train, w) ## function returns best threshold and associed precision, recall, f1-score, accuracy

    print(f"Best threshold: {x[0]:.2f}")
    print("Results:")
    print(f"  Precision {x[1]:.2f}")
    print(f"  Recall {x[2]:.2f}")
    print(f"  F1-Score {x[3]:.2f}")
    print(f"  Accuracy {x[4]:.2f}")


    # 4. Run model on test data and produce submission txt

    y_pred = Model.predict_logistic(w, tx_test, x[0])
    y_pred = np.where(y_pred == 1, 1, -1)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print("submission.csv created !")


    
    # 5. Create plot with decision boundary
    """
     age_range = np.arange(0, 81)
     prob_curve = impl.sigmoid(w[0] + w[1] * (age_range / 80))
    
     decision_age = (np.log(best_threshold / (1 - best_threshold)) - w[0]) / w[1] * 80
    
     ages = tx_train[:, 1] * 80
     plt.scatter(ages[y_train == 0], y_train[y_train == 0], alpha=0.01, s=5, color="steelblue", label="No disease")
     plt.scatter(ages[y_train == 1], y_train[y_train == 1], alpha=0.01, s=5, color="salmon",    label="Disease")
     plt.plot(age_range, prob_curve, color="black", linewidth=2, label="P(disease | age)")
     plt.axvline(decision_age, color="red", linestyle="--", label=f"Decision boundary (age={decision_age:.1f})")
     plt.ylim(-0.1, 1.1)
     plt.xlabel("Age")
     plt.ylabel("P(heart disease = 1)")
     plt.legend()
     plt.savefig("decision_boundary.png", dpi=150)
     plt.show()
    
    """
if __name__ == "__main__":
    main()
