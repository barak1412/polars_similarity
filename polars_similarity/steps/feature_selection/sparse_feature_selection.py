from typing import List, Dict, Any, Tuple
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from polars_similarity.steps.feature_selection.base_sparse_feature_selector import BaseSparseFeatureSelector
from polars_similarity.steps.feature_selection.model_sparse_feature_selector import ModelSparseFeatureSelector


_DEFAULT_MODEL_N_JOBS = 10
_TFIDF_LABELS_MAPPING_ATTR = 'tfidf_labels_mapping'


class SparseFeatureSelection(object):
    """
    Sparse features selection implementation for multiple sparse features.

    Parameters
    ----------
    features_metadata: Dict[str, Dict[str, Any]]
        Metadata of the given features in the form of dictionary.
        For example,
            {
                'feature1': {
                    'feature1_col':{
                        'tfidf_labels_mapping': 0: 'l0', 1: 'l3'}
                    }
                }
            }

    features_cols: List[str], Optional
        List of sparse features we want to apply selection on.
        If not give, will be applied to all of them.

    sparse_feature_selector: BaseSparseFeatureSelector, Optional
        Strategy for how to perform the actual selection.

    Returns
    -------
    Tuple of the new features dataframe after selection, and the updated features metadata.
    We have to update the features metadata because the mapping indices can be changed in tfidf based features columns.
    """
    def __init__(self, features_metadata: Dict[str, Dict[str, Any]],
                 features_cols: List[str] = None,
                 sparse_feature_selector: BaseSparseFeatureSelector = None):
        self._features_metadata = features_metadata
        self._features_cols = features_cols

        # in case feature selector is not supplied, we use random forest as default
        if sparse_feature_selector is None:
            estimator = RandomForestClassifier(n_jobs=_DEFAULT_MODEL_N_JOBS)
            sparse_feature_selector = ModelSparseFeatureSelector(estimator=estimator)
        self._sparse_feature_selector = sparse_feature_selector

    def process(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        # in case of empty features dataframe, we do nothing
        if X.shape[0] == 0:
            return X, self._features_metadata

        # we only work on the subset columns the user asked
        features_cols = self._features_cols if self._features_cols is not None else X.columns

        # iterate on every feature column for every feature
        resulted_features_metadata = self._features_metadata.copy()
        resulted_X = X.copy()
        for feature_name in self._features_metadata:
            for feature_col in self._features_metadata[feature_name]:
                # we skip non sparse or non relevant columns (columns the user didn't want)
                if feature_col in features_cols and SparseFeatureSelection._is_sparse_column(X, feature_col):
                    resulted_X[feature_col], feature_indices_mapping = self._select_from_sparse_feature(X[feature_col], y)

                    # in case we deal with tfidf feature, we want to update its labels mapping
                    if _TFIDF_LABELS_MAPPING_ATTR in self._features_metadata[feature_name][feature_col]:
                        old_labels_mapping = resulted_features_metadata[feature_name][feature_col][_TFIDF_LABELS_MAPPING_ATTR]
                        resulted_features_metadata[feature_name][feature_col][_TFIDF_LABELS_MAPPING_ATTR] = \
                                SparseFeatureSelection._get_updated_labels_mapping(old_labels_mapping, feature_indices_mapping)

        return resulted_X, resulted_features_metadata

    @staticmethod
    def _is_sparse_column(X: pd.DataFrame, feature_col: str) -> bool:
        first_row_feature_value = X.iloc[0][feature_col]

        return isinstance(first_row_feature_value, sp.csr_matrix)

    @staticmethod
    def _get_updated_labels_mapping(old_labels_mapping: Dict[int, Any], indices_mapping: Dict[int, int]) -> Dict[int, Any]:
        new_labels_mapping = {}
        for old_index, label in old_labels_mapping.items():
            if old_index in indices_mapping:
                new_labels_mapping[indices_mapping[old_index]] = label

        return new_labels_mapping

    def _select_from_sparse_feature(self, sparse_feature_series: pd.Series, y: pd.Series) -> Tuple[pd.Series, Dict[int, int]]:
        # choose what indices we want to take
        indices_taken_mask = self._sparse_feature_selector.select_indices(sparse_feature_series, y)

        # create mapping from the old sparse indices to the new sparse indices based on the chosen indices
        old_to_new_indices_mapping = {}
        current_index = 0
        for i in range(len(indices_taken_mask)):
            if indices_taken_mask[i]:
                old_to_new_indices_mapping[i] = current_index
                current_index += 1

        # recompute the sparse feature column
        resulted_sparse_feature_series = sparse_feature_series.apply(lambda csr_matrix: \
                                    SparseFeatureSelection._reindex_csr_matrix(csr_matrix, old_to_new_indices_mapping))

        return resulted_sparse_feature_series, old_to_new_indices_mapping

    @staticmethod
    def _reindex_csr_matrix(old_csr_matrix: sp.csr_matrix, indices_mapping: Dict[int, int]) -> sp.csr_matrix:
        new_indices = []
        new_values = []
        dim = len(indices_mapping)

        # we convert every old index to the new one based on mapping
        for idx, sparse_vector_index in enumerate(old_csr_matrix.indices):
            # we ignore all indices that were not selected (and as a result not in the mapping)
            if sparse_vector_index in indices_mapping:
                new_indices.append(indices_mapping[sparse_vector_index])
                new_values.append(old_csr_matrix.data[idx])
        new_csr_matrix = sp.csr_matrix((new_values, ([0] * len(new_indices), new_indices)), shape=(1, dim))

        return new_csr_matrix