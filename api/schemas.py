from typing import Literal

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float
    rank: int
    strategy: Literal["matrix_factorization", "popular_fallback"]


class RecommendResponse(BaseModel):
    user_id: int
    strategy: Literal["matrix_factorization", "popular_fallback"]
    known_user: bool
    recommendations: list[RecommendationItem]


class PopularResponse(BaseModel):
    strategy: Literal["popular_fallback"]
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    data_dir: str
    n_users: int
    n_movies: int
    n_known_model_users: int
    n_known_model_items: int


class ModelInfoResponse(BaseModel):
    model_name: str
    artifact_path: str
    n_known_model_users: int
    n_known_model_items: int
    supports_personalized_recommendation: bool
    supports_unknown_user_fallback: bool
    fallback_strategy: str


class ExplanationAnchor(BaseModel):
    movie_id: int
    title: str
    genres: str
    similarity: float


class ExplainResponse(BaseModel):
    user_id: int
    movie_id: int
    title: str
    genres: str
    known_user: bool
    known_item: bool
    strategy: Literal["matrix_factorization", "popular_fallback"]
    predicted_score: float | None
    popular_rank: int | None
    genre_overlap_count: int
    genre_overlap: list[str]
    supporting_movies: list[ExplanationAnchor]
    reason: str


class RecommendationQuery(BaseModel):
    top_k: int = Field(10, ge=1, le=100)
