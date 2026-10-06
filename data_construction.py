import os
import numpy as np

from github_workdir.helpers import load_csv_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def prepare_data(tx_train, tx_test, y_train, test_ids):
    tx_train = tx_train[:, [0, 249]]  # bias + _AGE80 (= age of a person, or 80 if age above 80)
    tx_test  = tx_test[:, [0, 249]]

    # Normalize age
    tx_train[:, 1] = tx_train[:, 1] / 80
    tx_test[:, 1]  = tx_test[:, 1] / 80

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

    return tx_train, tx_test, y_train, test_ids


# def prepare_data():
#     """
#     Drop obviously non-predictive columns and columns that are entirely NaN.
#     """
#     x_train, x_test, y_train, test_ids = load_csv_data_cached()
#
#     # Load feature names from the CSV header ("Id" is already stripped by the
#     # loader; the prepended bias term sits at index 0, so features start at 1)
#     with open(os.path.join(DATA_DIR, "x_train.csv"), "r") as f:
#         feature_names = f.readline().strip().split(",")[1:]  # drop "Id"
#
#     def indices(names):
#         return [feature_names.index(n) + 1 for n in names if n in feature_names]
#
#     # Interview timing — date/month/year/day of the phone call, not respondent health
#     TIMING = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR"]
#
#     # Survey administration — completion status, sequence number, sampling unit
#     ADMIN = ["DISPCODE", "SEQNO", "_PSU"]
#
#     # Landline-only call-routing questions — blank for all ~42% cell-phone
#     # respondents by design; encode survey logistics, not health status
#     LANDLINE_ROUTING = [
#         "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
#         "NUMADULT", "NUMMEN", "NUMWOMEN",
#     ]
#
#     # Cell-phone-only call-routing questions — blank for all ~58% landline
#     # respondents by design; encode survey logistics, not health status
#     CELLPHONE_ROUTING = [
#         "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS",
#         "CSTATE", "LANDLINE", "HHADULT",
#     ]
#
#     # Post-stratification survey weights — encode sampling probability for
#     # national representativeness, not any characteristic of the respondent
#     WEIGHTS = [
#         "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
#         "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT",
#     ]
#
#     # Questionnaire metadata — version and language of the survey instrument;
#     # reflect survey logistics rather than respondent health
#     QUESTIONNAIRE_META = ["QSTVER", "QSTLANG"]
#
#     DROP = TIMING + ADMIN + LANDLINE_ROUTING + CELLPHONE_ROUTING + WEIGHTS + QUESTIONNAIRE_META
#
#     keep = np.ones(x_train.shape[1], dtype=bool)
#     for idx in indices(DROP):
#         keep[idx] = False
#     x_train = x_train[:, keep]
#     x_test = x_test[:, keep]
#
#     # Drop columns never asked to any respondent (entirely NaN); use np.all so
#     # that columns with structural/conditional missingness are preserved
#     null_cols = np.all(np.isnan(x_train), axis=0)
#     x_train = x_train[:, ~null_cols]
#     x_test = x_test[:, ~null_cols]
#
#     return x_train, x_test, y_train, test_ids