import unittest
import polars as pl
from polars.testing import assert_frame_equal
from polars_similarity.steps.calculate_top_similarity_factory import CalculateTopCosineSimilarity


_LEFT_IDENTIFIER = 'id'
_RIGHT_IDENTIFIER = 'id'
_LEFT_DIM = 'dim'
_RIGHT_DIM = 'dim'
_LEFT_INDICES = 'indices'
_RIGHT_INDICES = 'indices'
_LEFT_VALUES = 'values'
_RIGHT_VALUES = 'values'


_OUTPUT_LEFT_IDENTIFIER = 'lid'
_OUTPUT_RIGHT_IDENTIFIER = 'rid'
_OUTPUT_K = 'k'
_OUTPUT_SCORE = 'score'


class TestCalculateTopCosineSimilarity(unittest.TestCase):

    def test_non_zero_vectors(self):
        # lid1 - [0.9, 5.3, 0, 0.4], lid2 - [0, 0.6, 0, 0]
        left_df = pl.DataFrame({
            _LEFT_IDENTIFIER: ['lid1', 'lid1', 'lid1', 'lid2'],
            _LEFT_DIM: [4, 4, 4, 4],
            _LEFT_INDICES: [0, 1, 3, 1],
            _LEFT_VALUES: [0.9, 5.3, 0.4, 0.6]
        })

        # rid1 - [0, 7.7, 0.3, 0.9], rid2 - [0, 4.6, 0, 0], rid3 - [0.8, 0, 0, 0]
        right_df = pl.DataFrame({
            _RIGHT_IDENTIFIER: ['rid1', 'rid1', 'rid1', 'rid2', 'rid3'],
            _RIGHT_DIM: [4, 4, 4, 4, 4],
            _RIGHT_INDICES: [1, 2, 3, 1, 0],
            _RIGHT_VALUES: [7.7, 0.3, 0.9, 4.6, 0.8]
        })

        cosine_similarity_calculator = CalculateTopCosineSimilarity(left_features_identifier_col=_LEFT_IDENTIFIER,
                            right_features_identifier_col=_RIGHT_IDENTIFIER, left_indices_col=_LEFT_INDICES,
                            right_indices_col=_RIGHT_INDICES, left_values_col=_LEFT_VALUES, right_values_col=_RIGHT_VALUES,
                            top_k=2, output_left_identifier_col=_OUTPUT_LEFT_IDENTIFIER,
                            output_right_identifier_col=_OUTPUT_RIGHT_IDENTIFIER, output_k_col=_OUTPUT_K, output_score_col=_OUTPUT_SCORE)
        result_df = cosine_similarity_calculator._process(left_features_lf=left_df.lazy(),
                                                          right_features_lf=right_df.lazy()).collect()
        # (lid1, rid1) -> - 0.9843, (lid1, rid2) -> - 0.9831, (lid1, rid3) -> - 0.1669
        # (lid2, rid1) -> - 0.9924, (lid2, rid2) -> - 1.0, (lid2, rid3) -> - 0
        expected_df = pl.DataFrame({
            _OUTPUT_LEFT_IDENTIFIER: ['lid1', 'lid1', 'lid2', 'lid2'],
            _OUTPUT_RIGHT_IDENTIFIER: ['rid1', 'rid2', 'rid1', 'rid2'],
            _OUTPUT_SCORE: [0.9842, 0.9831, 0.9924, 1.0],
            _OUTPUT_K: [1, 2, 2, 1]
        }).with_columns(pl.col(_OUTPUT_K).cast(pl.UInt32))

        assert_frame_equal(result_df, expected_df, check_row_order=False, check_column_order=False, check_exact=False,
                           atol=0.001)
