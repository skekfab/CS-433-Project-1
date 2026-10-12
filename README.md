# CS-433 Project 1 — Heart Disease Prediction

[![Grading Tests](https://github.com/skekfab/CS-433-Project-1/actions/workflows/grading.yml/badge.svg)](https://github.com/skekfab/CS-433-Project-1/actions/workflows/grading.yml)

**Authors:** Jonas Huber, Niklas Waibel, Ghalia Bennani

Binary classification on the BRFSS (Behavioral Risk Factor Surveillance System) dataset to predict whether a respondent has been diagnosed with coronary heart disease or myocardial infarction.

## Repository structure

| File                   | Description                                                                |
|------------------------|----------------------------------------------------------------------------|
| `run.py`               | Entry point — runs the full pipeline and writes `submission.csv`           |
| `data_construction.py` | Data loading, feature engineering, and cross-validation splits             |
| `implementations.py`   | Required ML implementations (least squares, ridge, logistic regression, …) |
| `model.py`             | Prediction, threshold selection, and evaluation metrics                    |
| `helpers.py`           | Provided utilities: CSV loading, submission writer, k-fold index builder   |
| `plots.py`             | Plotting helpers used during experimentation                               |

## Setup

```zsh
# [Optional] if you're using pyenv
pyenv local 3.14
poetry env use $(which python3)

poetry install
```

### Data

The data files are not included due to size. Download them from the [course repo](https://github.com/epfml/ML_course/tree/main/projects/project1/data) and place the unzipped CSVs (`x_train.csv`, `x_test.csv`, `y_train.csv`) in the `data/` directory.

On the first run the CSVs are parsed and cached as `.npy` files to speed up subsequent runs.

## Running

```zsh
python run.py
```

This produces `submission.csv` in the project root.

## Pipeline

### 1. Data preparation

We tried three feature sets (selectable at the top of `run.py`):

- **Age only** (`prepare_data_only_age`) — baseline with a single feature.
- **Non-null features** (`prepare_data_all_non_null_features`) — drop columns containing any NaN, keeping ~40 complete features.
- **Median imputation** (`prepare_data_median_imputed`) ← *used for the final submission* — keep all features except non-informative ones, impute missing values with the column median.

In all cases we:
1. Drop non-predictive columns: survey routing flags, timing fields, administrative fields, and sample weights.
2. Prepend a bias column of ones.
3. Apply mean–std normalization (computed on the training set, applied to both splits).

### 2. Model

Regularized logistic regression trained with mini-batch stochastic gradient descent (`reg_logistic_regression_stochastic` in `implementations.py`). Labels are encoded as {0, 1} internally; the submission uses {−1, 1}.

### 3. Cross-validation

3-fold cross-validation over λ ∈ {0, 10e-3, 10e-2, 0.1, 0.2, 0.5} to select the regularization strength that minimizes mean validation log-loss.

Fixed hyperparameters: learning rate γ = 0.05, 3000 iterations, mini-batch size 100, seed 42.

### 4. Threshold selection

After fixing the best λ, we retrain on 4/5 of the data and sweep decision thresholds in [0.01, 0.99]. The threshold maximizing F1-score on the validation fold is used for the final predictions.
