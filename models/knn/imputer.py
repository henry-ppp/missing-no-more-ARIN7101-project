"""K-nearest-neighbor imputation of standardized numerical data."""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


class KNNModel:
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        values = X.to_numpy(dtype=float, copy=True)
        observed = ~np.isnan(values)
        if not observed.any(axis=0).all():
            raise ValueError("every column needs at least one observed value")
        model = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights,
                           metric="nan_euclidean")
        filled = model.fit_transform(values)
        filled[observed] = values[observed]
        return pd.DataFrame(filled, index=X.index, columns=X.columns)
