import numpy as np
import matplotlib.pyplot as plt
from helpers import create_csv_submission, build_k_indices
import implementations as impl
import data_construction as Data
import model as Model

def main():

    ########################################################################################################################################
    # 1. Finding the best hyperparameter lambda for regularized logistic regression using cross-validation and stochastic gradient descent
    ########################################################################################################################################

    tx_train, tx_test, y_train, test_ids = Data.prepare_data_only_age(*Data.load_csv_data_cached())
    tx_train, tx_test, y_train, test_ids = Data.prepare_data_all_non_null_features(*Data.load_csv_data_cached())
    tx_train, tx_test, y_train, test_ids = Data.prepare_data_median_imputed(*Data.load_csv_data_cached())
    SEED = 42
    K_FOLD = 3
    MAX_ITERS = 5000
    GAMMA = 0.05
    INITIAL_W = np.zeros(tx_train.shape[1])
    ##INITIAL_W = np.random.uniform(-1, 1, tx_train.shape[1])
    MINIBATCH = 20
    LAMBDAS = [0, 0.01, 0.02]
    k_indices = build_k_indices(y_train, K_FOLD, SEED)

    loss_averages = []
    for lambda_index, lambda_ in enumerate(LAMBDAS):
        losses = []
        for k in range(0, K_FOLD):
            tx_tr , y_tr , tx_val , y_val = Data.data_cross_validated(tx_train, y_train, k_indices, k)
            w, _ = impl.reg_logistic_regression_stochastic(y_tr, tx_tr, lambda_, INITIAL_W, MAX_ITERS, GAMMA, MINIBATCH)
            log_loss_val = impl.logistic_loss(tx_val, y_val, w)
            losses.append(log_loss_val)
        loss_averages.append(np.mean(losses))

    ########################################################################################################################################
    # 2. Select best hyperparameter based on loss and retrain model on whole train dataset and plot training and validation loss over iterations
    ########################################################################################################################################

    best_lambda_index = np.argmin(loss_averages)

    print("=== LOSS AVERAGES ===")
    for lam, loss in zip(LAMBDAS, loss_averages):
        print(f"  lambda = {lam:.4f}  →  loss = {loss:.6f}")

    best_lambda = LAMBDAS[best_lambda_index]
    print(f"Best lambda: {best_lambda}")
    w, _ = impl.reg_logistic_regression_stochastic(y_train, tx_train, best_lambda, INITIAL_W, MAX_ITERS, GAMMA, MINIBATCH)

    ########################################################################################################################################
    # 3. Find decision threshold and evaluate model
    ########################################################################################################################################

    best_threshold, precision, recall, f1_score, accuracy = Model.best_threshold(tx_train, y_train, w) ## function returns best threshold and associated precision, recall, f1-score, accuracy
    print(f"Best threshold: {best_threshold:.2f}")
    print("Results:")
    print(f"  Precision {precision:.4f}")
    print(f"  Recall {recall:.4f}")
    print(f"  F1-Score {f1_score:.4f}")
    print(f"  Accuracy {accuracy:.4f}")

    ########################################################################################################################################
    # 4. Run model on test data and produce submission txt
    ########################################################################################################################################

    y_pred = Model.predict_logistic(w, tx_test, best_threshold)
    y_pred = np.where(y_pred == 1, 1, -1)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print("submission.csv created !")

    ########################################################################################################################################
    # 5. Plot training and validation loss over iterations for both regularized and non-regularized logistic regression
    ########################################################################################################################################

    k_indices = build_k_indices(y_train, 5, SEED) ; MINIBATCH = 100 ; INITIAL_W = np.zeros(tx_train.shape[1])
    tx_tr , y_tr , tx_val , y_val = Data.data_cross_validated(tx_train, y_train, k_indices, 0)
    tx_train_subset, y_train_subset , tx_tr , y_tr= tx_tr[:2000] , y_tr[:2000] , tx_tr[:10000] , y_tr[:10000]

    train_losses , val_losses , train_losses_reg , val_losses_reg , train_losses_all , val_losses_all , train_losses_all_reg , val_losses_all_reg = [] , [] , [] , [] , [] , [] , [] , []
    w , w_reg , w_all , w_all_reg = INITIAL_W , INITIAL_W , INITIAL_W , INITIAL_W

    for i in range(MAX_ITERS):

        ## Train model on 10k data and comparing train loss and validation loss with and without regularization

        w_all , loss_tr_all = impl.reg_logistic_regression_stochastic(y_tr, tx_tr, 0, w_all, 1, GAMMA, MINIBATCH)
        w_all_reg , loss_tr_all_reg = impl.reg_logistic_regression_stochastic(y_tr, tx_tr, 0.2, w_all_reg, 1, GAMMA, MINIBATCH)

        loss_vall_all = impl.logistic_loss(tx_val, y_val, w_all) ; loss_vall_all_reg = impl.logistic_loss(tx_val, y_val, w_all_reg)
        train_losses_all.append(loss_tr_all) ; train_losses_all_reg.append(loss_tr_all_reg)
        val_losses_all.append(loss_vall_all); val_losses_all_reg.append(loss_vall_all_reg)

        ## Train model on subset of data (2k) and comparing train loss and validation loss with and without regularization

        w , loss_tr = impl.reg_logistic_regression_stochastic(y_train_subset, tx_train_subset, 0, w, 1, GAMMA, MINIBATCH)
        w_reg , loss_tr_reg = impl.reg_logistic_regression_stochastic(y_train_subset, tx_train_subset, 0.2, w_reg, 1, GAMMA, MINIBATCH)

        loss_val = impl.logistic_loss(tx_val, y_val, w) ; loss_val_reg = impl.logistic_loss(tx_val, y_val, w_reg)
        train_losses.append(loss_tr) ; train_losses_reg.append(loss_tr_reg)
        val_losses.append(loss_val) ; val_losses_reg.append(loss_val_reg)

        if i % 1000 == 0:
            print(f"Iteration {i}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5)) ; fig2 , axes2 = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Validation')
    axes[0].set_title(f'Subset of data without regularization : $\\lambda$ = {0}')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Log-loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(train_losses_reg, label='Train')
    axes[1].plot(val_losses_reg, label='Validation')
    axes[1].set_title(f'Subset of data with regularization : $\\lambda$ = {0.2}')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Log-loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes2[0].plot(train_losses_all, label='Train')
    axes2[0].plot(val_losses_all, label='Validation')
    axes2[0].set_title(f'10k datas without regularization : $\\lambda$ = {0}')
    axes2[0].set_xlabel('Iteration')
    axes2[0].set_ylabel('Log-loss')
    axes2[0].legend()
    axes2[0].grid(alpha=0.3)

    axes2[1].plot(train_losses_all_reg, label='Train')
    axes2[1].plot(val_losses_all_reg, label='Validation')
    axes2[1].set_title(f'10k datas with regularization : $\\lambda$ = {0.2}')
    axes2[1].set_xlabel('Iteration')
    axes2[1].set_ylabel('Log-loss')
    axes2[1].legend()
    axes2[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
