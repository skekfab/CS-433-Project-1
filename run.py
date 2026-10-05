import numpy as np
import matplotlib.pyplot as plt
from helpers import load_csv_data, create_csv_submission
import implementations as impl
import os
import Data_construction as Data

def main():
  
    tx_train, tx_test, y_train, test_ids = Data.load_cached()

    for i in range(tx_train.shape[1]):
        print("NaN totaux :", np.isnan(tx_train[i]).sum())

    N = len(y_train)
    indices = np.random.permutation(N)
    split = int(N * 0.8)

    X_train , Y_train = tx_train[indices[:split]] , y_train[indices[:split]]
    X_val , Y_val = tx_train[indices[split:]] , y_train[indices[split:]]
    
    N , D = tx_train.shape
    gamma , lambda_ = 0.01, 0.01

    w_init = np.zeros(D)
    w = w_init
    loss_train , loss_val = [], []
    
    for i in range(1000):
        loss_train.append(impl.reg_logistic_regression_stochastic(Y_train, X_train, lambda_, w, 1 , gamma)[1])
        loss_val.append(impl.reg_logistic_regression_stochastic(Y_val, X_val, lambda_, w, 0 , gamma)[1])
        ##print("loss_val[i] =", loss_train[i])
        w = impl.reg_logistic_regression_stochastic(Y_train, X_train, lambda_, w, 1 , gamma)[0]

    fig, axes = plt.subplots(figsize = (12,5))
    axes.plot(range(len(loss_train)), loss_train, label='Training Loss')
    axes.plot(range(len(loss_val)), loss_val, label='Validation Loss')
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$\mathcal{L}$')
    axes.grid(True, alpha=0.3)
    plt.show()













    print("tx_train.shape =", tx_train.shape)
    print("tx_train.shape")
    lambda_ = 0.1
    w, loss = impl.ridge_regression(y_train, tx_train, lambda_)
    print(f"Loss: {loss}")

    y_pred_raw = tx_test @ w
    y_pred = np.where(y_pred_raw > 0, 1, -1)

    create_csv_submission(test_ids, y_pred, "submission.csv")
    print("submission.csv créé !")

if __name__ == "__main__":
    main()