from pathlib import Path

import pandas as pd

from src.config import MOVIES_COLUMNS, RATINGS_COLUMNS, RAW_DIR, USERS_COLUMNS


def _read_dat(path: Path, columns: list[str], encoding: str = "latin-1") -> pd.DataFrame:
    return pd.read_csv(path, sep="::", names=columns, engine="python", encoding=encoding)


def load_users(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or RAW_DIR
    users = _read_dat(base / "users.dat", USERS_COLUMNS)
    return users


def load_movies(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or RAW_DIR
    movies = _read_dat(base / "movies.dat", MOVIES_COLUMNS)
    return movies


def load_ratings(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or RAW_DIR
    ratings = _read_dat(base / "ratings.dat", RATINGS_COLUMNS)
    return ratings


def load_movielens_1m(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = load_users(data_dir)
    movies = load_movies(data_dir)
    ratings = load_ratings(data_dir)
    return users, movies, ratings
