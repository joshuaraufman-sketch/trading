from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a copy of df containing a boolean 'signal' column.
        """
        raise NotImplementedError