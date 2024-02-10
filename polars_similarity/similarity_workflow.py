import polars as pl
from polars_similarity.steps.sparse_features_assembler import SparseFeaturesAssembler
from polars_similarity.steps.calculate_top_similarity_factory import CalculateTopSimilarityFactory
from polars_similarity.constants import INDICES, VALUES


class SimilarityWorkflow(object):
    def __init__(self, features_identifier_col: str, top_k: int = 10,
                 normalize_features: bool = True):
        self._features_identifier_col = features_identifier_col
        self._normalize_features = normalize_features
        self._top_k = top_k

    def _process(self, features_lf: pl.LazyFrame):
        # merge features to one column
        final_merged_sparse_feature_col = 'final_merged_sparse_feature_col'
        assembler = SparseFeaturesAssembler(identifier_col=self._features_identifier_col,
                                            output_col=final_merged_sparse_feature_col)
        merged_features_lf = assembler._process(features_lf)

        # transform the sparse features column to tabular form
        output_indices_col = 'indices'
        output_values_col = 'values'
        tabular_features_lf = self._convert_sparse_feature_to_tabular(merged_features_lf,
                                                                      final_merged_sparse_feature_col,
                                                                      output_values_col=output_values_col,
                                                                      output_indices_col=output_indices_col)

        # normalize features if needed
        if self._normalize_features:
            tabular_features_lf = SimilarityWorkflow._normalize_tabular_features(tabular_features_lf,
                                                                                 indices_col=output_indices_col,
                                                                                 values_col=output_values_col)
        # create similarity calculator (we use 'cosine' as metric)
        similarity_calculator = CalculateTopSimilarityFactory.create(
            left_features_identifier_col=self._features_identifier_col,
            right_features_identifier_col=self._features_identifier_col,
            left_indices_col=output_indices_col, right_indices_col=output_indices_col,
            left_values_col=output_values_col,
            right_values_col=output_values_col, top_k=self._top_k, metric='cosine',
            output_left_identifier_col='gt_id',
            output_right_identifier_col='pool_id',
            output_score_col='score',
            output_k_col='k',
        )

        similarity_lf = similarity_calculator._process(tabular_features_lf, tabular_features_lf)

        return similarity_lf

    def _convert_sparse_feature_to_tabular(self, features_lf: pl.LazyFrame,
                                           sparse_features_col: str, output_indices_col: str,
                                           output_values_col: str) -> pl.LazyFrame:
        tabular_features_lf = features_lf.select(
            pl.col(self._features_identifier_col),
            pl.col(sparse_features_col).struct.field(INDICES).alias(output_indices_col),
            pl.col(sparse_features_col).struct.field(VALUES).alias(output_values_col)
        ).explode([output_indices_col, output_values_col])

        return tabular_features_lf

    @staticmethod
    def _normalize_tabular_features(features_lf: pl.LazyFrame, indices_col: str, values_col: str):
        # we use 'l2' normalization
        normalized_features_lf = features_lf.with_columns(
            pl.col(values_col) / (pl.col(values_col) ** 2).sum().sqrt() \
            .over([indices_col]).alias(values_col)
        )

        return normalized_features_lf
