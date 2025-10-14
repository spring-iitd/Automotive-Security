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
        """Apply the attack to the given data."""
        pass


class EvasionAttack(Attack):
    @abc.abstractmethod
    def apply(self, frames: list[dict], labels: np.ndarray | None = None, **kwargs) -> list[dict]:
        """Apply evasion attack to the given frames."""
        pass

class StatisticalAttack(Attack):
    @abc.abstractmethod
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply statistical attack to the given DataFrame."""
        pass
