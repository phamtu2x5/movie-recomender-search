import os
from pathlib import Path


def _resolve_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_artifacts_dir(project_root: Path) -> Path:
    # Colab workflow:
    # - code and raw data run from /content/<repo-name>
    # - AI artifacts are persisted back to Drive under /content/drive/MyDrive/<repo-name>/AI
    drive_repo = Path("/content/drive/MyDrive") / project_root.name
    if str(project_root).startswith("/content/") and drive_repo.exists():
        return drive_repo / "AI" / "artifacts"
    return project_root / "AI" / "artifacts"


ROOT_DIR = _resolve_path("MOVIE_REC_PROJECT_ROOT", _default_project_root())
DATA_DIR = _resolve_path("MOVIE_REC_DATA_DIR", ROOT_DIR / "data")
RAW_DIR = _resolve_path("MOVIE_REC_RAW_DIR", DATA_DIR / "raw" / "ml-1m")
INTERIM_DIR = _resolve_path("MOVIE_REC_INTERIM_DIR", DATA_DIR / "interim")
PROCESSED_DIR = _resolve_path("MOVIE_REC_PROCESSED_DIR", DATA_DIR / "processed")
ARTIFACTS_DIR = _resolve_path("MOVIE_REC_ARTIFACTS_DIR", _default_artifacts_dir(ROOT_DIR))
MODELS_DIR = _resolve_path("MOVIE_REC_MODELS_DIR", ARTIFACTS_DIR / "models")
METRICS_DIR = _resolve_path("MOVIE_REC_METRICS_DIR", ARTIFACTS_DIR / "metrics")
FIGURES_DIR = _resolve_path("MOVIE_REC_FIGURES_DIR", ARTIFACTS_DIR / "figures")


RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]
MOVIES_COLUMNS = ["movie_id", "title", "genres"]


DEFAULT_RANDOM_STATE = 42
DEFAULT_TOP_K = 10
