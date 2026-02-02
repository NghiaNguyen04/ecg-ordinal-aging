from imblearn.over_sampling import ADASYN
from sklearn.utils import shuffle
import numpy as np

class ADASYNWrapper:
    """
    ADASYN oversampling for numpy ndarray data.

    Parameters
    ----------
    sampling_strategy : float, str, dict or callable, default='auto'
        Sampling strategy for ADASYN.
    n_neighbors : int or object, default=5
        Number of nearest neighbours to use for synthetic samples.
    random_state : int, default=42
        Random seed for reproducibility.
    integer_indices : list[int], optional
        Indices of columns in X that must remain integer-valued after resampling.
    """

    def __init__(self, sampling_strategy='auto', n_neighbors=5, random_state=42, use_bmi=False, use_sex = False):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.use_bmi = use_bmi
        self.use_sex = use_sex
        self.adasyn = ADASYN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state
        )

    def fit_resample(self, X, y):
        """
        Perform ADASYN oversampling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        X_resampled : np.ndarray
        y_resampled : np.ndarray
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        X_res, y_res = self.adasyn.fit_resample(X, y)

        # Làm tròn các cột cần là số nguyên
        if self.use_sex:
            X_res[:, -1] = np.round(X_res[:, -1]).astype(int)

        X_res, y_res = shuffle(X_res, y_res, random_state=self.random_state)

        return X_res, y_res
