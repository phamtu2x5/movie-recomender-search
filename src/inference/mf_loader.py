from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import MODELS_DIR


@dataclass
class MFArtifactBundle:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    global_bias: float
    user_ids: np.ndarray
    item_ids: np.ndarray

    def __post_init__(self) -> None:
        self.user_id_to_index = {int(user_id): idx for idx, user_id in enumerate(self.user_ids.tolist())}
        self.item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(self.item_ids.tolist())}

    @classmethod
    def from_npz(cls, path: str | Path) -> "MFArtifactBundle":
        bundle = np.load(Path(path), allow_pickle=True)
        return cls(
            user_factors=bundle["user_factors"],
            item_factors=bundle["item_factors"],
            user_bias=bundle["user_bias"],
            item_bias=bundle["item_bias"],
            global_bias=float(bundle["global_bias"]),
            user_ids=bundle["user_ids"],
            item_ids=bundle["item_ids"],
        )

    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_id_to_index

    def has_item(self, movie_id: int) -> bool:
        return movie_id in self.item_id_to_index

    def known_item_ids(self) -> list[int]:
        return [int(item_id) for item_id in self.item_ids.tolist()]

    def predict_single(self, user_id: int, movie_id: int) -> float:
        scores = self.predict_for_user(user_id, [movie_id])
        if len(scores) == 0:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        return float(scores[0])

    def item_similarity(self, left_movie_id: int, right_movie_id: int) -> float:
        if not self.has_item(left_movie_id):
            raise KeyError(f"Unknown movie_id: {left_movie_id}")
        if not self.has_item(right_movie_id):
            raise KeyError(f"Unknown movie_id: {right_movie_id}")

        left_idx = self.item_id_to_index[left_movie_id]
        right_idx = self.item_id_to_index[right_movie_id]
        left_vector = self.item_factors[left_idx]
        right_vector = self.item_factors[right_idx]
        denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
        if denominator == 0:
            return 0.0
        return float(np.dot(left_vector, right_vector) / denominator)

    def predict_for_user(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        if not self.has_user(user_id):
            raise KeyError(f"Unknown user_id: {user_id}")

        valid_movie_ids = [movie_id for movie_id in movie_ids if movie_id in self.item_id_to_index]
        if not valid_movie_ids:
            return np.asarray([], dtype=float)

        user_idx = self.user_id_to_index[user_id]
        item_indices = np.asarray([self.item_id_to_index[movie_id] for movie_id in valid_movie_ids], dtype=int)
        user_vector = self.user_factors[user_idx]

        scores = (
            self.global_bias
            + self.user_bias[user_idx]
            + self.item_bias[item_indices]
            + self.item_factors[item_indices] @ user_vector
        )
        return np.clip(scores, 1.0, 5.0)


def load_mf_artifact(path: str | Path | None = None) -> MFArtifactBundle:
    model_path = Path(path) if path is not None else MODELS_DIR / "mf_model.npz"
    return MFArtifactBundle.from_npz(model_path)
