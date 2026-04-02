import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.inference.recommender import MovieRecommenderService

    parser = argparse.ArgumentParser(description="Show top-N movie recommendations for a user.")
    parser.add_argument("user_id", type=int, help="MovieLens user_id")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return")
    args = parser.parse_args()

    service = MovieRecommenderService()
    payload = service.recommend(user_id=args.user_id, top_k=args.top_k)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
