"""SoftImpute (low-rank) and batch Sinkhorn (optimal transport) imputation. See README.md.

    from models.softimpute_sinkhorn import impute_softimpute, impute_sinkhorn
    filled, info = impute_softimpute(X_missing, seed=0, fraction=0.05)   # NumPy array with NaN gaps
    filled, info = impute_sinkhorn(X_missing, seed=0)
"""
from .imputers import impute_sinkhorn, impute_softimpute, soft_impute, soft_impute_path

__all__ = ['impute_sinkhorn', 'impute_softimpute', 'soft_impute', 'soft_impute_path']
