import pandas as pd


def random_split_explicit(
    ratings: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0")

    shuffled = ratings.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_total = len(shuffled)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train = shuffled.iloc[:n_train].copy()
    val = shuffled.iloc[n_train : n_train + n_val].copy()
    test = shuffled.iloc[n_train + n_val :].copy()
    return train, val, test


def leave_one_out_split(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split: last item for test, previous for validation."""
    ordered = ratings.sort_values([user_col, time_col]).copy()

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in ordered.groupby(user_col):
        if len(group) < 3:
            train_parts.append(group)
            continue
        train_parts.append(group.iloc[:-2])
        val_parts.append(group.iloc[-2:-1])
        test_parts.append(group.iloc[-1:])

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True) if val_parts else ordered.iloc[0:0].copy()
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else ordered.iloc[0:0].copy()
    return train, val, test
