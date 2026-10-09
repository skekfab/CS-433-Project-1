import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype, ndarray
from typing import Any
import implementations as impl


########################################################################################################################################
# Visualize logistic regression curve and decision boundary on age feature
########################################################################################################################################
def age_only_plot(seed: int, best_threshold: float, tx_train, w: ndarray[tuple[Any, ...], dtype[Any]], y_train):
    fig, ax = plt.subplots(figsize=(9, 5))

    age_train = tx_train[:, 1] * 62 + 18  # de-normalize: [0, 1] → [0, 80]

    rng = np.random.default_rng(seed)
    idx_vis = rng.choice(len(age_train), size=min(3000, len(age_train)), replace=False)
    ax.scatter(age_train[idx_vis], y_train[idx_vis],
               alpha=0.2, s=6, color='steelblue', label='Data points')

    age_range = np.linspace(0, 80, 500)
    p_pred = impl.sigmoid(w[0] + w[1] * (age_range / 80))
    ax.plot(age_range, p_pred, color='darkorange', linewidth=2.5,
            label=r'Logistic regression')

    ax.axhline(best_threshold, color='red', linestyle='--', linewidth=1.5, label='Decision threshold')

    if w[1] != 0 and 0 < best_threshold < 1:
        logit_val = np.log(best_threshold / (1 - best_threshold))
        age_db = (logit_val - w[0]) / w[1] * 80
        if 0 <= age_db <= 80:
            ax.axvline(age_db, color='green', linestyle=':', linewidth=1.5, label='Decision boundary')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('P(disease = 1)')
    ax.set_title('Logistic regression using age only')
    ax.set_xlim(18, 80)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("age_only.pdf")
    plt.show()