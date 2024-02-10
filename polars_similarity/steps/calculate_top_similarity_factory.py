from polars_similarity.steps.base_calculate_top_similarity import BaseCalculateTopSimilarity
from polars_similarity.steps.calculate_top_cosine_similarity import CalculateTopCosineSimilarity

_SUPPORTED_METRICS_CLASSES = {
    'cosine': CalculateTopCosineSimilarity
}


class CalculateTopSimilarityFactory(object):
    @staticmethod
    def create(left_features_identifier_col: str,
               right_features_identifier_col: str,
               left_indices_col: str,
               right_indices_col: str,
               left_values_col: str,
               right_values_col: str,
               top_k: int,
               output_left_identifier_col: str,
               output_right_identifier_col: str,
               output_score_col: str,
               output_k_col: str,
               metric: str) -> BaseCalculateTopSimilarity:
        # validate metric is supported
        if metric not in _SUPPORTED_METRICS_CLASSES:
            raise Exception(f'Unsupported metric {metric} for similarity.')

        similarity_obj = _SUPPORTED_METRICS_CLASSES[metric](
            left_features_identifier_col=left_features_identifier_col,
            right_features_identifier_col=right_features_identifier_col,
            left_indices_col=left_indices_col,
            right_indices_col=right_indices_col,
            left_values_col=left_values_col,
            right_values_col=right_values_col,
            top_k=top_k,
            output_left_identifier_col=output_left_identifier_col,
            output_right_identifier_col=output_right_identifier_col,
            output_score_col=output_score_col,
            output_k_col=output_k_col
        )

        return similarity_obj
