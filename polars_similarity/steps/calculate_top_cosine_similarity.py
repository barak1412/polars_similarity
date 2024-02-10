import polars as pl
from polars_similarity.steps.base_calculate_top_similarity import BaseCalculateTopSimilarity


class CalculateTopCosineSimilarity(BaseCalculateTopSimilarity):
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
        super().__init__(left_features_identifier_col=left_features_identifier_col,
                         right_features_identifier_col=right_features_identifier_col,
                         left_indices_col=left_indices_col, right_indices_col=right_indices_col,
                         left_values_col=left_values_col,
                         right_values_col=right_values_col, top_k=top_k,
                         output_left_identifier_col=output_left_identifier_col,
                         output_right_identifier_col=output_right_identifier_col, output_score_col=output_score_col,
                         output_k_col=output_k_col)

    def _process(self, left_features_lf: pl.LazyFrame, right_features_lf: pl.LazyFrame) -> pl.LazyFrame:
        # rename columns to avoid collisions and perform normalization for the cosine similarity
        left_indices_col = f'left_{self._left_indices_col}'
        right_indices_col = f'right_{self._right_indices_col}'
        left_values_col = f'left_{self._left_values_col}'
        right_values_col = f'right_{self._right_values_col}'
        normalized_left_features_lf = CalculateTopCosineSimilarity._prepare_features(left_features_lf,
                                                                                     identifier_col=self._left_features_identifier_col,
                                                                                     output_identifier_col=self._output_left_identifier_col,
                                                                                     indices_col=self._left_indices_col,
                                                                                     output_indices_col=left_indices_col,
                                                                                     values_col=self._left_values_col,
                                                                                     output_values_col=left_values_col)
        normalized_right_features_lf = CalculateTopCosineSimilarity._prepare_features(right_features_lf,
                                                                                      identifier_col=self._right_features_identifier_col,
                                                                                      output_identifier_col=self._output_right_identifier_col,
                                                                                      indices_col=self._right_indices_col,
                                                                                      output_indices_col=right_indices_col,
                                                                                      values_col=self._right_values_col,
                                                                                      output_values_col=right_values_col)

        # perform matrix multiplication
        multi_score = 'mult_score'
        matrix_lf = normalized_left_features_lf.join(normalized_right_features_lf,
                                                     left_on=left_indices_col, right_on=right_indices_col, how='inner') \
            .with_columns((pl.col(left_values_col) * pl.col(right_values_col)).alias(multi_score))

        # sum values foreach pair (left_id, right_id) to get the final score
        final_score_lf = matrix_lf.group_by([self._output_left_identifier_col, self._output_right_identifier_col]) \
            .agg(pl.sum(multi_score).alias(self._output_score_col)) \
            .filter(pl.col(self._output_left_identifier_col) != pl.col(self._output_right_identifier_col))

        # take only top K
        final_score_ranked_lf = final_score_lf.with_columns(
            pl.col(self._output_score_col).rank(method='ordinal', descending=True).over(
                self._output_left_identifier_col).alias(self._output_k_col)
        ).filter(pl.col(self._output_k_col) <= self._top_k)

        return final_score_ranked_lf

    @staticmethod
    def _prepare_features(features_lf: pl.LazyFrame, identifier_col: str, output_identifier_col: str,
                          indices_col: str, output_indices_col: str, values_col,
                          output_values_col: str) -> pl.LazyFrame:
        # change columns names
        processed_features_lf = features_lf.select(pl.col(identifier_col).alias(output_identifier_col),
                                                   pl.col(indices_col).alias(output_indices_col),
                                                   pl.col(values_col).alias(output_values_col))

        # normalize the feature foreach identifier
        processed_features_lf = processed_features_lf.with_columns(
            pl.col(output_values_col) / (pl.col(output_values_col) ** 2).sum().sqrt() \
            .over([output_identifier_col]).alias(output_values_col)
        )

        return processed_features_lf
