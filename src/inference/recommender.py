from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RAW_DIR
from src.data.load_movielens import load_movies, load_ratings
from src.inference.mf_loader import MFArtifactBundle, load_mf_artifact


@dataclass
class RecommendationResult:
    movie_id: int
    title: str
    genres: str
    score: float
    rank: int
    strategy: str


@dataclass
class ExplanationAnchor:
    movie_id: int
    title: str
    genres: str
    similarity: float


class MovieRecommenderService:
    def __init__(
        self,
        model_bundle: MFArtifactBundle | None = None,
        model_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else RAW_DIR
        self.model_path = Path(model_path) if model_path is not None else MODELS_DIR / "mf_model.npz"
        self.model = model_bundle or load_mf_artifact(self.model_path)

        self.movies = load_movies(self.data_dir).copy()
        self.ratings = load_ratings(self.data_dir).copy()
        self.movies_by_id = self.movies.set_index("movie_id")
        self.seen_by_user = self.ratings.groupby("user_id")["movie_id"].agg(set).to_dict()
        self.popular_movies = self._build_popularity_ranking()
        self.popular_rank_by_movie = {
            int(movie_id): rank
            for rank, movie_id in enumerate(self.popular_movies["movie_id"].tolist(), start=1)
        }

    def _build_popularity_ranking(self) -> pd.DataFrame:
        movie_stats = (
            self.ratings.groupby("movie_id")
            .agg(
                mean_rating=("rating", "mean"),
                rating_count=("rating", "count"),
            )
            .reset_index()
        )

        global_mean = float(movie_stats["mean_rating"].mean())
        min_votes = float(movie_stats["rating_count"].quantile(0.75))
        votes = movie_stats["rating_count"]
        means = movie_stats["mean_rating"]

        movie_stats["weighted_score"] = (
            (votes / (votes + min_votes)) * means
            + (min_votes / (votes + min_votes)) * global_mean
        )
        ranked = movie_stats.merge(self.movies, on="movie_id", how="left")
        ranked = ranked.sort_values(["weighted_score", "rating_count"], ascending=[False, False]).reset_index(drop=True)
        return ranked

    def health(self) -> dict:
        return {
            "status": "ok",
            "data_dir": str(self.data_dir),
            "n_users": int(self.ratings["user_id"].nunique()),
            "n_movies": int(self.movies["movie_id"].nunique()),
            "n_known_model_users": int(len(self.model.user_ids)),
            "n_known_model_items": int(len(self.model.item_ids)),
        }

    def model_info(self) -> dict:
        return {
            "model_name": "matrix_factorization",
            "artifact_path": str(self.model_path),
            "n_known_model_users": int(len(self.model.user_ids)),
            "n_known_model_items": int(len(self.model.item_ids)),
            "supports_personalized_recommendation": True,
            "supports_unknown_user_fallback": True,
            "fallback_strategy": "popular_fallback",
        }

    def recommend(self, user_id: int, top_k: int = 10) -> dict:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.model.has_user(user_id):
            recommendations = self._recommend_known_user(user_id=user_id, top_k=top_k)
        else:
            recommendations = self._recommend_popular(top_k=top_k)

        strategy = recommendations[0].strategy if recommendations else "popular_fallback"

        return {
            "user_id": user_id,
            "strategy": strategy,
            "known_user": self.model.has_user(user_id),
            "recommendations": [result.__dict__ for result in recommendations],
        }

    def popular(self, top_k: int = 10) -> dict:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        recommendations = self._recommend_popular(top_k=top_k)
        return {
            "strategy": "popular_fallback",
            "recommendations": [result.__dict__ for result in recommendations],
        }

    def explain_recommendation(self, user_id: int, movie_id: int) -> dict:
        if movie_id not in self.movies_by_id.index:
            raise ValueError(f"Unknown movie_id: {movie_id}")

        movie = self.movies_by_id.loc[movie_id]
        is_known_user = self.model.has_user(user_id)
        is_known_item = self.model.has_item(movie_id)
        popular_rank = self.popular_rank_by_movie.get(movie_id)
        genre_tokens = self._genre_tokens(str(movie["genres"]))

        explanation = {
            "user_id": user_id,
            "movie_id": int(movie_id),
            "title": str(movie["title"]),
            "genres": str(movie["genres"]),
            "known_user": is_known_user,
            "known_item": is_known_item,
            "strategy": "matrix_factorization" if is_known_user and is_known_item else "popular_fallback",
            "predicted_score": None,
            "popular_rank": popular_rank,
            "genre_overlap_count": 0,
            "genre_overlap": [],
            "supporting_movies": [],
            "reason": "",
        }

        if not is_known_user:
            explanation["reason"] = (
                "User chưa có trong mô hình nên hệ thống dùng fallback theo độ phổ biến của phim."
            )
            return explanation

        seen_movie_ids = list(self.seen_by_user.get(user_id, set()))
        seen_known_movie_ids = [seen_movie_id for seen_movie_id in seen_movie_ids if self.model.has_item(seen_movie_id)]

        if is_known_item:
            explanation["predicted_score"] = round(self.model.predict_single(user_id, movie_id), 4)

        supporting_movies = self._supporting_movies(movie_id=movie_id, seen_movie_ids=seen_known_movie_ids, top_k=3)
        explanation["supporting_movies"] = [anchor.__dict__ for anchor in supporting_movies]

        overlap = self._genre_overlap_with_history(target_genres=genre_tokens, seen_movie_ids=seen_movie_ids)
        explanation["genre_overlap_count"] = len(overlap)
        explanation["genre_overlap"] = sorted(overlap)

        if supporting_movies:
            explanation["reason"] = (
                "Phim này được gợi ý vì vector ẩn của nó gần với các phim người dùng đã xem/rate trước đó."
            )
        elif is_known_item:
            explanation["reason"] = (
                "Phim này được gợi ý vì mô hình Matrix Factorization dự đoán điểm phù hợp cao cho người dùng này."
            )
        else:
            explanation["reason"] = (
                "Phim này không nằm trong tập item của mô hình, nên chỉ có thể giải thích bằng độ phổ biến và thể loại."
            )
        return explanation

    def _recommend_known_user(self, user_id: int, top_k: int) -> list[RecommendationResult]:
        seen_movie_ids = self.seen_by_user.get(user_id, set())
        candidate_movie_ids = [movie_id for movie_id in self.model.known_item_ids() if movie_id not in seen_movie_ids]
        scores = self.model.predict_for_user(user_id, candidate_movie_ids)

        if len(scores) == 0:
            return self._recommend_popular(top_k=top_k)

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_movie_ids = [candidate_movie_ids[idx] for idx in top_indices]
        top_scores = scores[top_indices]

        results: list[RecommendationResult] = []
        for rank, (movie_id, score) in enumerate(zip(top_movie_ids, top_scores), start=1):
            movie = self.movies_by_id.loc[movie_id]
            results.append(
                RecommendationResult(
                    movie_id=int(movie_id),
                    title=str(movie["title"]),
                    genres=str(movie["genres"]),
                    score=round(float(score), 4),
                    rank=rank,
                    strategy="matrix_factorization",
                )
            )
        return results

    def _recommend_popular(self, top_k: int) -> list[RecommendationResult]:
        results: list[RecommendationResult] = []
        for rank, row in enumerate(self.popular_movies.head(top_k).itertuples(index=False), start=1):
            results.append(
                RecommendationResult(
                    movie_id=int(row.movie_id),
                    title=str(row.title),
                    genres=str(row.genres),
                    score=round(float(row.weighted_score), 4),
                    rank=rank,
                    strategy="popular_fallback",
                )
            )
        return results

    def _genre_tokens(self, genres: str) -> set[str]:
        if not genres:
            return set()
        return {genre.strip() for genre in genres.split("|") if genre.strip()}

    def _genre_overlap_with_history(self, target_genres: set[str], seen_movie_ids: list[int]) -> set[str]:
        overlap: set[str] = set()
        for seen_movie_id in seen_movie_ids:
            if seen_movie_id not in self.movies_by_id.index:
                continue
            seen_genres = self._genre_tokens(str(self.movies_by_id.loc[seen_movie_id]["genres"]))
            overlap.update(target_genres.intersection(seen_genres))
        return overlap

    def _supporting_movies(
        self,
        movie_id: int,
        seen_movie_ids: list[int],
        top_k: int,
    ) -> list[ExplanationAnchor]:
        if not self.model.has_item(movie_id):
            return []

        anchors: list[tuple[int, float]] = []
        for seen_movie_id in seen_movie_ids:
            if seen_movie_id == movie_id:
                continue
            similarity = self.model.item_similarity(movie_id, seen_movie_id)
            anchors.append((seen_movie_id, similarity))

        anchors.sort(key=lambda item: item[1], reverse=True)
        results: list[ExplanationAnchor] = []
        for supporting_movie_id, similarity in anchors[:top_k]:
            if supporting_movie_id not in self.movies_by_id.index:
                continue
            movie = self.movies_by_id.loc[supporting_movie_id]
            results.append(
                ExplanationAnchor(
                    movie_id=int(supporting_movie_id),
                    title=str(movie["title"]),
                    genres=str(movie["genres"]),
                    similarity=round(float(similarity), 4),
                )
            )
        return results
