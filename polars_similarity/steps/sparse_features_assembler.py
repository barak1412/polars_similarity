from typing import List
import polars as pl
from polars_similarity.constants import DIM, INDICES, VALUES


class SparseFeaturesAssembler(object):
    def __init__(self, identifier_col: str, output_col: str, features_cols = None):
        self._identifier_col = identifier_col
        self._output_col = output_col
        self._features_cols = features_cols

    def _process(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        assembled_cols = self._features_cols
        if assembled_cols is None:
            assembled_cols = [col for col in lf.columns if col != self._identifier_col]

        # in case we don't have columns to merge, we do nothing
        if len(assembled_cols) == 0:
            return lf
        final_sparse_cols = []

        # assemble all numeric, non vectors (Lists or Structs) to one sparse column
        sparse_numeric_col = 'sparse_numeric_col'
        numeric_cols = [col for col in assembled_cols if lf.schema[col] not in [pl.List, pl.Struct]]
        if len(numeric_cols) > 0:
            final_sparse_cols.append(sparse_numeric_col)

        # transform all List column to sparse (Struct)
        list_cols = [col for col in assembled_cols if lf.schema[col] == pl.List]
        final_sparse_cols.extend(list_cols)

        # merge all sparse columns to one sparse column
        struct_cols = [col for col in assembled_cols if lf.schema[col] == pl.Struct]
        final_sparse_cols.extend(struct_cols)
        lf = self._merge_all_sparse_columns(lf, sparse_cols=final_sparse_cols)

        return lf

    def _merge_all_sparse_columns(self, lf: pl.LazyFrame, sparse_cols: List[str]):
        first_sparse_col = sparse_cols[0]
        for iter_sparse_col in sparse_cols[1:]:
            lf = SparseFeaturesAssembler._merge_two_sparse_columns(lf,
                                                                   first_col=first_sparse_col,
                                                                   second_col=iter_sparse_col,
                                                                   output_col=first_sparse_col)
        lf = lf.rename({first_sparse_col: self._output_col})

        return lf

    @staticmethod
    def _merge_two_sparse_columns(lf: pl.LazyFrame, first_col: str, second_col: str, output_col: str):
        temp_row_index = 'temp_row_index'
        result_lf = lf.with_row_index(temp_row_index).with_columns(
            pl.struct(
                (pl.col(first_col).struct.field(DIM) + pl.col(second_col).struct.field(DIM)).alias(DIM),
                pl.concat_list(pl.col(first_col).struct.field(INDICES),
                               (
                                   (pl.col(second_col).struct.field(INDICES).list.explode()
                                    + pl.col(first_col).struct.field(DIM)).implode().over(temp_row_index)
                               )).alias(INDICES),
                pl.concat_list(pl.col(first_col).struct.field(VALUES),
                               pl.col(second_col).struct.field(VALUES)).alias(VALUES)
            ).alias(output_col)
        ).drop(temp_row_index)

        # drop original columns
        if first_col != output_col:
            result_lf = result_lf.drop(first_col)
        if second_col != output_col:
            result_lf = result_lf.drop(second_col)

        return result_lf
