import pandas as pd


def normalize_explicit_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    frame = ratings.copy()
    frame["rating_normalized"] = frame["rating"] / 5.0
    return frame


def build_implicit_feedback(ratings: pd.DataFrame) -> pd.DataFrame:
    frame = ratings[["user_id", "movie_id", "timestamp"]].copy()
    frame["label"] = 1
    return frame


def extract_release_year(movies: pd.DataFrame) -> pd.DataFrame:
    frame = movies.copy()
    frame["release_year"] = frame["title"].str.extract(r"\((\d{4})\)")
    return frame
