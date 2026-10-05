import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def load_cached():
    """Charge les données avec biais + y en 0/1."""
    x_train  = np.load(os.path.join(DATA_DIR, "x_train.npy"))
    x_test   = np.load(os.path.join(DATA_DIR, "x_test.npy"))
    y_train  = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    test_ids = np.load(os.path.join(DATA_DIR, "test_ids.npy"))

    tx_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    tx_test  = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    y_train  = (y_train > 0).astype(int)

    return tx_train, tx_test, y_train, test_ids

print("Data_construction.py loaded")