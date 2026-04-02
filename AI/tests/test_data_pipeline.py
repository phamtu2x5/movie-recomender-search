import pandas as pd

from AI.src.data.split import random_split_explicit


def test_random_split_preserves_row_count() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "movie_id": [10, 11, 12, 13, 14],
            "rating": [5, 4, 3, 4, 5],
            "timestamp": [1, 2, 3, 4, 5],
        }
    )

    train, val, test = random_split_explicit(frame, train_frac=0.6, val_frac=0.2, random_state=42)
    assert len(train) + len(val) + len(test) == len(frame)
