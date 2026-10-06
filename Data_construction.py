import os
import numpy as np

from github_workdir.helpers import load_csv_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def load_cached():
    """Charge les données avec biais + y en 0/1."""
    try:
        x_train  = np.load(os.path.join(DATA_DIR, "x_train.npy"))
        x_test   = np.load(os.path.join(DATA_DIR, "x_test.npy"))
        y_train  = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        test_ids = np.load(os.path.join(DATA_DIR, "test_ids.npy"))
    except FileNotFoundError:
        print("Cached numpy data does not exist, loading from csv. This may take some time.")
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)
        np.save(os.path.join(DATA_DIR, "x_train.npy"), x_train)
        np.save(os.path.join(DATA_DIR, "x_test.npy"), x_test)
        np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(DATA_DIR, "train_ids.npy"), train_ids)
        np.save(os.path.join(DATA_DIR, "test_ids.npy"), test_ids)
        print("Loading finished")

    tx_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    tx_test  = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    y_train  = (y_train > 0).astype(int)

    return tx_train, tx_test, y_train, test_ids
