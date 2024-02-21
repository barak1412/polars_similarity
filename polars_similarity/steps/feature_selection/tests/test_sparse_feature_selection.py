import unittest
from typing import List

import pandas as pd
import scipy.sparse as sp
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from polars_similarity.steps.feature_selection.sparse_feature_selection import SparseFeatureSelection, _TFIDF_LABELS_MAPPING_ATTR
from polars_similarity.steps.feature_selection.base_sparse_feature_selector import BaseSparseFeatureSelector
from polars_similarity.steps.feature_selection.model_sparse_feature_selector import ModelSparseFeatureSelector


_IDENTIFIER = 'id'
_SPARSE_COL = 'sparse'
_DENSE_COL = 'dense'
_NUMERIC_COL = 'numeric'


class StubSparseFeatureSelector(BaseSparseFeatureSelector):

    def select_indices(self, sparse_feature_series: pd.Series, y: pd.Series) -> List[bool]:
        return [True, False, False, True, False, False]


class TestSparseFeatureSelection(unittest.TestCase):
    @staticmethod
    def _csr_matrix_col_to_list(X: pd.DataFrame, sparse_col: str) -> pd.DataFrame:
        new_X = X.copy()
        new_X[sparse_col] = new_X[sparse_col].apply(lambda sparse_matrix: sparse_matrix.toarray()[0].tolist())

        return new_X

    def test_one_sparse_column(self):
        X = pd.DataFrame({
            _IDENTIFIER: ['id1', 'id2', 'id3'],
            _SPARSE_COL: [sp.csr_matrix(([8.0, 4.0], ([0, 0], [0, 4])), shape=(1, 6)),
                          sp.csr_matrix(([9.3, 1.5, 0.7], ([0, 0, 0], [0, 2, 3])), shape=(1, 6)),
                          sp.csr_matrix(([0.7, 5.4], ([0, 0], [1, 5])), shape=(1, 6))]
        }).set_index(_IDENTIFIER)

        y = pd.Series(data=['id1', 'id2', 'id3'], index=['id1', 'id2', 'id3'])

        features_metadata = {
            _SPARSE_COL: {
                _SPARSE_COL: {
                    _TFIDF_LABELS_MAPPING_ATTR: {0: 'l0', 1: 'l1', 2: 'l2', 3: 'l3', 4: 'l4', 5: 'l5'}
                }
            }
        }

        sparse_feature_selection = SparseFeatureSelection(features_metadata=features_metadata,
                                                           sparse_feature_selector=StubSparseFeatureSelector())
        result_X, result_metadata = sparse_feature_selection.process(X, y)
        expected_X = pd.DataFrame({
            _IDENTIFIER: ['id1', 'id2', 'id3'],
            _SPARSE_COL: [sp.csr_matrix(([8.0], ([0], [0])), shape=(1, 2)),
                          sp.csr_matrix(([9.3, 0.7], ([0, 0], [0, 1])), shape=(1, 2)),
                          sp.csr_matrix(([], ([], [])), shape=(1, 2))]
        }).set_index(_IDENTIFIER)

        expected_metadata = {
            _SPARSE_COL: {
                _SPARSE_COL: {
                    _TFIDF_LABELS_MAPPING_ATTR: {0: 'l0', 1: 'l3'}
                }
            }
        }

        # workaround because csr_matrix can not be compared
        result_X = TestSparseFeatureSelection._csr_matrix_col_to_list(result_X, _SPARSE_COL)
        expected_X = TestSparseFeatureSelection._csr_matrix_col_to_list(expected_X, _SPARSE_COL)

        self.assertDictEqual(expected_metadata, result_metadata)
        self.assertDictEqual(expected_X.to_dict(), result_X.to_dict())

    def test_all_column_types(self):
        X = pd.DataFrame({
            _IDENTIFIER: ['id1', 'id2', 'id3'],
            _SPARSE_COL: [sp.csr_matrix(([8.0, 4.0], ([0, 0], [0, 4])), shape=(1, 6)),
                          sp.csr_matrix(([9.3, 1.5, 0.7], ([0, 0, 0], [0, 2, 3])), shape=(1, 6)),
                          sp.csr_matrix(([0.7, 5.4], ([0, 0], [1, 5])), shape=(1, 6))],
            _DENSE_COL: [[0, 9], [8, 6], [3, 3]],
            _NUMERIC_COL: [0.7, 0.0, 7.4]
        }).set_index(_IDENTIFIER)

        y = pd.Series(data=['id1', 'id2', 'id3'], index=['id1', 'id2', 'id3'])

        features_metadata = {
            _SPARSE_COL: {
                _SPARSE_COL: {
                    _TFIDF_LABELS_MAPPING_ATTR: {0: 'l0', 1: 'l1', 2: 'l2', 3: 'l3', 4: 'l4', 5: 'l5'}
                }
            }
        }

        sparse_feature_selection = SparseFeatureSelection(features_metadata=features_metadata,
                                                           sparse_feature_selector=StubSparseFeatureSelector())
        result_X, result_metadata = sparse_feature_selection.process(X, y)
        expected_X = pd.DataFrame({
            _IDENTIFIER: ['id1', 'id2', 'id3'],
            _SPARSE_COL: [sp.csr_matrix(([8.0], ([0], [0])), shape=(1, 2)),
                          sp.csr_matrix(([9.3, 0.7], ([0, 0], [0, 1])), shape=(1, 2)),
                          sp.csr_matrix(([], ([], [])), shape=(1, 2))],
            _DENSE_COL: [[0, 9], [8, 6], [3, 3]],
            _NUMERIC_COL: [0.7, 0.0, 7.4]
        }).set_index(_IDENTIFIER)

        expected_metadata = {
            _SPARSE_COL: {
                _SPARSE_COL: {
                    _TFIDF_LABELS_MAPPING_ATTR: {0: 'l0', 1: 'l3'}
                }
            }
        }

        # workaround because csr_matrix can not be compared
        result_X = TestSparseFeatureSelection._csr_matrix_col_to_list(result_X, _SPARSE_COL)
        expected_X = TestSparseFeatureSelection._csr_matrix_col_to_list(expected_X, _SPARSE_COL)

        self.assertDictEqual(expected_metadata, result_metadata)
        self.assertDictEqual(expected_X.to_dict(), result_X.to_dict())

    def test_with_model(self):
        X = pd.DataFrame({
            _IDENTIFIER: ['id1', 'id2', 'id3'],
            _SPARSE_COL: [sp.csr_matrix(([8.0, 4.0], ([0, 0], [0, 4])), shape=(1, 6)),
                          sp.csr_matrix(([9.3, 0.7, 1.5], ([0, 0, 0], [0, 1, 2])), shape=(1, 6)),
                          sp.csr_matrix(([0.7, 5.4], ([0, 0], [1, 5])), shape=(1, 6))]
        }).set_index(_IDENTIFIER)

        y = pd.Series(data=['id1', 'id2', 'id3'], index=['id1', 'id2', 'id3'])

        features_metadata = {
            _SPARSE_COL: {
                _SPARSE_COL: {
                    _TFIDF_LABELS_MAPPING_ATTR: {0: 'l0', 1: 'l1', 2: 'l2', 3: 'l3', 4: 'l4', 5: 'l5'}
                }
            }
        }

        sparse_feature_selection = SparseFeatureSelection(features_metadata=features_metadata,
                        sparse_feature_selector=ModelSparseFeatureSelector(estimator=RandomForestClassifier(random_state=random.seed(42))))
        result_X, result_metadata = sparse_feature_selection.process(X, y)
        expected_X = pd.DataFrame({
            _IDENTIFIER: ['id1', 'id2', 'id3'],
            _SPARSE_COL: [sp.csr_matrix(([8.0, 4.0], ([0, 0], [0, 1])), shape=(1, 3)),
                          sp.csr_matrix(([9.3], ([0], [0])), shape=(1, 3)),
                          sp.csr_matrix(([5.4], ([0], [2])), shape=(1, 3))]
        }).set_index(_IDENTIFIER)

        expected_metadata = {
            _SPARSE_COL: {
                _SPARSE_COL: {
                    _TFIDF_LABELS_MAPPING_ATTR: {0: 'l0', 1: 'l4', 2: 'l5'}
                }
            }
        }

        # workaround because csr_matrix can not be compared
        result_X = TestSparseFeatureSelection._csr_matrix_col_to_list(result_X, _SPARSE_COL)
        expected_X = TestSparseFeatureSelection._csr_matrix_col_to_list(expected_X, _SPARSE_COL)

        self.assertDictEqual(expected_metadata, result_metadata)
        self.assertDictEqual(expected_X.to_dict(), result_X.to_dict())