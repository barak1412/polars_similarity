from abc import abstractmethod
import polars as pl


class BaseCalculateTopSimilarity(object):
    def __init__(self, left_features_identifier_col: str,
                 right_features_identifier_col: str,
                 left_indices_col: str,
                 right_indices_col: str,
                 left_values_col: str,
                 right_values_col: str,
                 top_k: int,
                 output_left_identifier_col: str,
                 output_right_identifier_col: str,
                 output_score_col: str,
                 output_k_col: str):
        self._left_features_identifier_col = left_features_identifier_col
        self._right_features_identifier_col = right_features_identifier_col
        self._left_indices_col = left_indices_col
        self._right_indices_col = right_indices_col
        self._left_values_col = left_values_col
        self._right_values_col = right_values_col
        self._top_k = top_k
        self._output_score_col = output_score_col
        self._output_k_col = output_k_col

        # verify output left and right identifiers are not equal
        self._output_left_identifier_col = output_left_identifier_col
        self._output_right_identifier_col = output_right_identifier_col

    @abstractmethod
    def _process(self, left_features_lf: pl.LazyFrame, right_features_lf: pl.LazyFrame) -> pl.LazyFrame:
        pass
