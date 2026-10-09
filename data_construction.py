import os
import numpy as np

from helpers import load_csv_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def prepare_data_only_age(tx_train, tx_test, y_train, test_ids):
    tx_train = tx_train[:, [0, 249]]  # bias + _AGE80 (= age of a person, or 80 if age above 80)
    tx_test  = tx_test[:, [0, 249]]

    # Normalize age
    tx_train[:, 1] = (tx_train[:, 1] - 18) / 62
    tx_test[:, 1]  = (tx_test[:, 1] - 18) / 62

    return tx_train, tx_test, y_train, test_ids


def prepare_data_all_non_null_features(tx_train, tx_test, y_train, test_ids):
    #
    # 0. Helper functions
    #

    # Load feature names / index from csv
    with open(os.path.join(DATA_DIR, "x_train.csv"), "r") as f:
        feature_names = f.readline().strip().split(",")[1:]  # drop "Id"

    # Translate name to index since we're using numpy and not pandas
    def indices(names):
        return [feature_names.index(n) + 1 for n in names if n in feature_names]

    #
    # 1. Drop columns with no predictive power
    #
    TIMING = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR"]
    ADMIN = ["DISPCODE", "SEQNO", "_PSU"]
    LANDLINE_ROUTING = ["CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT", "NUMADULT", "NUMMEN",
                        "NUMWOMEN"]
    CELLPHONE_ROUTING = ["CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT"]
    WEIGHTS = ["_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT"]
    QUESTIONNAIRE_META = ["QSTVER", "QSTLANG"]
    DROP = TIMING + ADMIN + LANDLINE_ROUTING + CELLPHONE_ROUTING + WEIGHTS + QUESTIONNAIRE_META

    keep = np.ones(tx_train.shape[1], dtype=bool)
    for idx in indices(DROP):
        keep[idx] = False
    tx_train = tx_train[:, keep]
    tx_test = tx_test[:, keep]

    #
    # 2. Only keep columns with no nans and do mean std normalization
    #
    non_null = ~np.any(np.isnan(tx_train[:, 1:]), axis=0)
    col_indices = np.concatenate([[0], np.where(non_null)[0] + 1])

    tx_train = tx_train[:, col_indices]
    tx_test  = tx_test[:, col_indices]

    means = tx_train[:, 1:].mean(axis=0)
    stds = tx_train[:, 1:].std(axis=0)
    tx_train[:, 1:] = (tx_train[:, 1:] - means) / stds
    tx_test[:, 1:]  = (tx_test[:, 1:]  - means) / stds

    return tx_train, tx_test, y_train, test_ids


def prepare_data_median_imputed(tx_train, tx_test, y_train, test_ids):
    with open(os.path.join(DATA_DIR, "x_train.csv"), "r") as f:
        feature_names = f.readline().strip().split(",")[1:]

    def indices(names):
        return [feature_names.index(n) + 1 for n in names if n in feature_names]

    TIMING = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR"]
    ADMIN = ["DISPCODE", "SEQNO", "_PSU"]
    LANDLINE_ROUTING = ["CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT", "NUMADULT", "NUMMEN", "NUMWOMEN"]
    CELLPHONE_ROUTING = ["CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT"]
    WEIGHTS = ["_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT"]
    QUESTIONNAIRE_META = ["QSTVER", "QSTLANG"]
    DROP = TIMING + ADMIN + LANDLINE_ROUTING + CELLPHONE_ROUTING + WEIGHTS + QUESTIONNAIRE_META

    keep = np.ones(tx_train.shape[1], dtype=bool)
    for idx in indices(DROP):
        keep[idx] = False
    tx_train = tx_train[:, keep]
    tx_test  = tx_test[:, keep]

    all_null = np.all(np.isnan(tx_train[:, 1:]), axis=0)
    col_indices = np.concatenate([[0], np.where(~all_null)[0] + 1])
    tx_train = tx_train[:, col_indices]
    tx_test  = tx_test[:, col_indices]

    medians = np.nanmedian(tx_train[:, 1:], axis=0)
    tx_train[:, 1:] = np.where(np.isnan(tx_train[:, 1:]), medians, tx_train[:, 1:])
    tx_test[:, 1:]  = np.where(np.isnan(tx_test[:, 1:]),  medians, tx_test[:, 1:])

    means = tx_train[:, 1:].mean(axis=0)
    stds  = tx_train[:, 1:].std(axis=0)
    stds[stds == 0] = 1
    tx_train[:, 1:] = (tx_train[:, 1:] - means) / stds
    tx_test[:, 1:]  = (tx_test[:, 1:]  - means) / stds

    return tx_train, tx_test, y_train, test_ids


def load_csv_data_cached():
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

    # Switch the labels from {-1, 1} to {0, 1}
    y_train = np.where(y_train == 1, 1, 0)

    return tx_train, tx_test, y_train, test_ids


def data_cross_validated(tx_train, y_train, k_indices, k):
    tr_indices = np.concatenate([k_indices[j] for j in range(len(k_indices)) if j != k])
    val_indices =  k_indices[k]
    tx_tr = tx_train[tr_indices]
    y_tr = y_train[tr_indices]
    tx_val = tx_train[val_indices]
    y_val = y_train[val_indices]
    return tx_tr, y_tr, tx_val, y_val