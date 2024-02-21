from polars_similarity.steps.feature_selection.base_sparse_feature_selector import BaseSparseFeatureSelector
from typing import List
import pandas as pd
import scipy as sp
from sklearn.feature_selection import SelectFromModel


class ModelSparseFeatureSelector(BaseSparseFeatureSelector):
    def __init__(self, estimator: object, threshold='mean'):
        self._estimator = estimator
        self._threshold = threshold

    def select_indices(self, sparse_feature_series: pd.Series, y: pd.Series) -> List[bool]:
        # transform the series of csr_matrix to one csr matrix
        feature_csr_matrix = sp.vstack(sparse_feature_series)

        # use estimator to get support
        model_feature_selector = SelectFromModel(self._estimator, threshold=self._threshold)
        model_feature_selector.fit(feature_csr_matrix, y)
        indices_support = model_feature_selector.get_support()

        return list(indices_support)