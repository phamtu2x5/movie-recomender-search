import pandas as pd


def build_id_mappings(ratings: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    user_ids = sorted(ratings["user_id"].unique().tolist())
    item_ids = sorted(ratings["movie_id"].unique().tolist())
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {movie_id: idx for idx, movie_id in enumerate(item_ids)}
    return user_to_index, item_to_index


def apply_id_mappings(
    frame: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> pd.DataFrame:
    mapped = frame.copy()
    mapped["user_idx"] = mapped["user_id"].map(user_to_index)
    mapped["item_idx"] = mapped["movie_id"].map(item_to_index)
    mapped = mapped.dropna(subset=["user_idx", "item_idx"]).copy()
    mapped["user_idx"] = mapped["user_idx"].astype(int)
    mapped["item_idx"] = mapped["item_idx"].astype(int)
    return mapped


def explode_genres(movies: pd.DataFrame) -> pd.DataFrame:
    frame = movies.copy()
    frame["genre"] = frame["genres"].str.split("|")
    return frame.explode("genre")
