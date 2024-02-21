from abc import abstractmethod, ABC
from typing import List
import pandas as pd


class BaseSparseFeatureSelector(ABC):
    @abstractmethod
    def select_indices(self, sparse_feature_series: pd.Series, y: pd.Series) -> List[bool]:
        pass
