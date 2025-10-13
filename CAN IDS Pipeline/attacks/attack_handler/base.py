import abc
import numpy as np
from typing import Any, Union
import pandas as pd

class Attack(abc.ABC):
    attack_params: list[str] = []

    def __init__(self, **kwargs):
        self.params = kwargs

    @abc.abstractmethod
    def apply(self, **kwargs):
        """
        Core attack logic. The meaning of parameters is up to subclass: 
        could be frames, dataframes, model, training data, etc.
        """
        pass

    # ... (include set_params, is_estimator_valid, etc.)

class EvasionAttack(Attack):
    @abc.abstractmethod
    def apply(self, frames: list[dict], labels: np.ndarray | None = None, **kwargs) -> list[dict]:
        """
        Perturb actual CAN frames or features to cause model misclassification.
        """

class StatisticalAttack(Attack):
    @abc.abstractmethod
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Perturb traffic-level statistical features.
        """
