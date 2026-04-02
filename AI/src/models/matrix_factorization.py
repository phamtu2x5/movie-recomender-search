from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class MFConfig:
    num_factors: int = 32
    learning_rate: float = 0.01
    reg: float = 0.02
    epochs: int = 10
    random_state: int = 42
    show_progress: bool = True


class MatrixFactorization:
    """Simple SGD-based explicit matrix factorization baseline."""

    def __init__(self, num_users: int, num_items: int, config: MFConfig | None = None):
        self.config = config or MFConfig()
        rng = np.random.default_rng(self.config.random_state)

        self.user_factors = rng.normal(0.0, 0.1, size=(num_users, self.config.num_factors))
        self.item_factors = rng.normal(0.0, 0.1, size=(num_items, self.config.num_factors))
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_bias = 0.0

    def _validate_columns(self, ratings: pd.DataFrame) -> pd.DataFrame:
        required = {"user_idx", "item_idx", "rating"}
        missing = required.difference(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
        return ratings[["user_idx", "item_idx", "rating"]].copy()

    def initialize_global_bias(self, ratings: pd.DataFrame) -> None:
        frame = self._validate_columns(ratings)
        self.global_bias = float(frame["rating"].mean())

    def run_epoch(self, ratings: pd.DataFrame, epoch_idx: int) -> float:
        frame = self._validate_columns(ratings)
        shuffled = frame.sample(frac=1.0, random_state=self.config.random_state + epoch_idx).itertuples(index=False)
        iterator = shuffled
        if self.config.show_progress:
            iterator = tqdm(
                shuffled,
                total=len(frame),
                desc=f"MF Epoch {epoch_idx + 1}/{self.config.epochs}",
                leave=False,
            )

        squared_error = 0.0
        for user_idx, item_idx, rating in iterator:
            pred = self.predict_single(user_idx, item_idx)
            err = rating - pred
            squared_error += err**2

            p_u = self.user_factors[user_idx].copy()
            q_i = self.item_factors[item_idx].copy()

            self.user_bias[user_idx] += self.config.learning_rate * (
                err - self.config.reg * self.user_bias[user_idx]
            )
            self.item_bias[item_idx] += self.config.learning_rate * (
                err - self.config.reg * self.item_bias[item_idx]
            )
            self.user_factors[user_idx] += self.config.learning_rate * (
                err * q_i - self.config.reg * p_u
            )
            self.item_factors[item_idx] += self.config.learning_rate * (
                err * p_u - self.config.reg * q_i
            )

        return float(np.sqrt(squared_error / len(frame)))

    def get_state(self) -> dict:
        return {
            "user_factors": self.user_factors.copy(),
            "item_factors": self.item_factors.copy(),
            "user_bias": self.user_bias.copy(),
            "item_bias": self.item_bias.copy(),
            "global_bias": float(self.global_bias),
        }

    def set_state(self, state: dict) -> None:
        self.user_factors = state["user_factors"].copy()
        self.item_factors = state["item_factors"].copy()
        self.user_bias = state["user_bias"].copy()
        self.item_bias = state["item_bias"].copy()
        self.global_bias = float(state["global_bias"])

    def fit(self, ratings: pd.DataFrame) -> list[float]:
        frame = self._validate_columns(ratings)
        self.global_bias = float(frame["rating"].mean())

        history: list[float] = []
        for epoch_idx in range(self.config.epochs):
            history.append(self.run_epoch(frame, epoch_idx))
        return history

    def predict_single(self, user_idx: int, item_idx: int) -> float:
        score = (
            self.global_bias
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return float(score)

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        preds = [self.predict_single(u, i) for u, i in zip(user_indices, item_indices)]
        return np.asarray(preds, dtype=float)
