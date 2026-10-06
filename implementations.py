import numpy as np


def MSE(tx, y, w):
    N = tx.shape[0]
    e = y - tx @ w
    return (1 / (2 * N)) * (e.T @ e)


def MSE_gradient(tx, y, w):
    N = tx.shape[0]
    e = y - tx @ w
    return -1 / N * tx.T @ e


def MSE_stochastic_gradient(tx, y, w):
    N = tx.shape[0]
    i = np.random.randint(N)
    return MSE_gradient(tx[i : i + 1], y[i : i + 1], w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_gradient(tx, y, w):
    N = tx.shape[0]
    return 1 / N * tx.T @ (sigmoid(tx @ w) - y)


def logistic_loss(tx, y, w):
    N = tx.shape[0]
    g = tx @ w
    loss = -y * g - np.log(sigmoid(-g))
    return np.mean(loss)


def least_squares(y, tx):
    """Closed-form least squares."""
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w, MSE(tx, y, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression, returned loss etxcludes penalty."""
    N, d = tx.shape
    w = np.linalg.inv(tx.T @ tx + 2 * N * lambda_ * np.eye(d)) @ tx.T @ y
    return w, MSE(tx, y, w)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression via gradient descent on MSE."""
    w = initial_w
    for i in range(max_iters):
        w = w - gamma * MSE_gradient(tx, y, w)
    return w, MSE(tx, y, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression via stochastic gradient descent on MSE."""
    w = initial_w
    for i in range(max_iters):
        w = w - gamma * MSE_stochastic_gradient(tx, y, w)
    return w, MSE(tx, y, w)


def logistic_regression(y, tx, initial_w, matx_iters, gamma):
    """Logistic regression via gradient descent."""
    w = initial_w
    for i in range(matx_iters):
        w = w - gamma * logistic_gradient(tx, y, w)
    return w, logistic_loss(tx, y, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, matx_iters, gamma):
    """Regularized logistic regression, returned loss etxcludes penalty."""
    w = initial_w
    for i in range(matx_iters):
        w = w - gamma * (logistic_gradient(tx, y, w) + 2 * lambda_ * w)
    return w, logistic_loss(tx, y, w)

def reg_logistic_regression_stochastic(y, tx, lambda_, initial_w, matx_iters, gamma):
    w = initial_w
    for i in range(matx_iters):
        N = tx.shape[0]
        idx = np.random.randint(N)
        w = w - gamma * (logistic_gradient(tx[idx:idx+1], y[idx:idx+1], w) + 2 * lambda_ * w)
    return w, logistic_loss(tx, y, w)
