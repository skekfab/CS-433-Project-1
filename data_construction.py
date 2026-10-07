import os
import numpy as np

from helpers import load_csv_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def prepare_data_only_age(tx_train, tx_test, y_train, test_ids):
    tx_train = tx_train[:, [0, 249]]  # bias + _AGE80 (= age of a person, or 80 if age above 80)
    tx_test  = tx_test[:, [0, 249]]

    # Normalize age
    tx_train[:, 1] = tx_train[:, 1] / 80
    tx_test[:, 1]  = tx_test[:, 1] / 80

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

    # Interview timing — date/month/year/day of the phone call, not respondent health
    TIMING = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR"]

    # Survey administration — completion status, sequence number, sampling unit
    ADMIN = ["DISPCODE", "SEQNO", "_PSU"]

    # Landline-only call-routing questions — blank for all ~42% cell-phone
    # respondents by design; encode survey logistics, not health status
    LANDLINE_ROUTING = [
        "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
        "NUMADULT", "NUMMEN", "NUMWOMEN",
    ]

    # Cell-phone-only call-routing questions — blank for all ~58% landline
    # respondents by design; encode survey logistics, not health status
    CELLPHONE_ROUTING = [
        "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS",
        "CSTATE", "LANDLINE", "HHADULT",
    ]

    # Post-stratification survey weights — encode sampling probability for
    # national representativeness, not any characteristic of the respondent
    WEIGHTS = [
        "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
        "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT",
    ]

    # Questionnaire metadata — version and language of the survey instrument;
    # reflect survey logistics rather than respondent health
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
