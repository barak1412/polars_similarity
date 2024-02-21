from abc import abstractmethod, ABC
from typing import List
import pandas as pd


class BaseSparseFeatureSelector(ABC):
    """
    Defines interface for single sparse feature selector.
    """
    @abstractmethod
    def select_indices(self, sparse_feature_series: pd.Series, y: pd.Series) -> List[bool]:
        """
        Parameters
        ----------
        sparse_feature_series: pd.Series,
            Pandas series that contains in each row scipy csr_matrix as sparse vector.

        y: pd.Series
            The labels (usually 0 or 1 for binary classifiers) of each row.

        Returns
        -------
        List of booleans that indicates for each dimension in the sparse vector whether we chosen to take him or not.
        """
        pass
