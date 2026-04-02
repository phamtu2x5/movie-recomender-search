from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from api.schemas import ExplainResponse, HealthResponse, ModelInfoResponse, PopularResponse, RecommendResponse
from src.inference.recommender import MovieRecommenderService


app = FastAPI(title="Movie Recommender API", version="0.1.0")


@lru_cache(maxsize=1)
def get_service() -> MovieRecommenderService:
    return MovieRecommenderService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    return HealthResponse(**service.health())


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    service = get_service()
    return ModelInfoResponse(**service.model_info())


@app.get("/popular", response_model=PopularResponse)
def popular(top_k: int = Query(10, ge=1, le=100)) -> PopularResponse:
    service = get_service()
    try:
        payload = service.popular(top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PopularResponse(**payload)


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, top_k: int = Query(10, ge=1, le=100)) -> RecommendResponse:
    service = get_service()
    try:
        payload = service.recommend(user_id=user_id, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RecommendResponse(**payload)


@app.get("/recommend/{user_id}/explain/{movie_id}", response_model=ExplainResponse)
def explain_recommendation(user_id: int, movie_id: int) -> ExplainResponse:
    service = get_service()
    try:
        payload = service.explain_recommendation(user_id=user_id, movie_id=movie_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExplainResponse(**payload)
