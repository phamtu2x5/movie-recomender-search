import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import METRICS_DIR, MODELS_DIR
from AI.src.evaluation.metrics import mae, rmse
from AI.src.features.encode import apply_id_mappings, build_id_mappings
from AI.src.models.matrix_factorization import MFConfig, MatrixFactorization


def _evaluate_explicit(
    model: MatrixFactorization,
    frame: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> dict:
    mapped = apply_id_mappings(frame, user_to_index, item_to_index)
    if mapped.empty:
        return {"rmse": None, "mae": None, "n_rows": 0}

    preds = model.predict(mapped["user_idx"].to_numpy(), mapped["item_idx"].to_numpy())
    preds = np.clip(preds, 1.0, 5.0)
    return {
        "rmse": rmse(mapped["rating"].to_numpy(), preds),
        "mae": mae(mapped["rating"].to_numpy(), preds),
        "n_rows": int(len(mapped)),
    }


def train_matrix_factorization(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame | None = None,
    config: MFConfig | None = None,
) -> tuple[MatrixFactorization, dict, dict[int, int], dict[int, int]]:
    config = config or MFConfig()
    user_to_index, item_to_index = build_id_mappings(train_ratings)

    train = apply_id_mappings(train_ratings, user_to_index, item_to_index)
    model = MatrixFactorization(len(user_to_index), len(item_to_index), config=config)
    model.initialize_global_bias(train)

    train_history: list[float] = []
    val_history_rmse: list[float] = []
    val_history_mae: list[float] = []
    best_val_rmse = float("inf")
    best_epoch = 0
    best_state = model.get_state()

    epoch_iterator = range(config.epochs)
    if config.show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc="MF training", leave=False)

    for epoch_idx in epoch_iterator:
        train_rmse = model.run_epoch(train, epoch_idx)
        val_metrics_epoch = _evaluate_explicit(model, val_ratings, user_to_index, item_to_index)

        train_history.append(train_rmse)
        val_history_rmse.append(val_metrics_epoch["rmse"])
        val_history_mae.append(val_metrics_epoch["mae"])

        if val_metrics_epoch["rmse"] is not None and val_metrics_epoch["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics_epoch["rmse"]
            best_epoch = epoch_idx + 1
            best_state = model.get_state()

        if config.show_progress and hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                train_rmse=round(train_rmse, 4),
                val_rmse=round(val_metrics_epoch["rmse"], 4),
            )

        print(
            f"Epoch {epoch_idx + 1:02d}/{config.epochs:02d} | "
            f"train_rmse={train_rmse:.4f} | "
            f"val_rmse={val_metrics_epoch['rmse']:.4f} | "
            f"val_mae={val_metrics_epoch['mae']:.4f}"
        )

    model.set_state(best_state)
    val_metrics = _evaluate_explicit(model, val_ratings, user_to_index, item_to_index)
    test_metrics = (
        _evaluate_explicit(model, test_ratings, user_to_index, item_to_index)
        if test_ratings is not None
        else None
    )
    results = {
        "model": "matrix_factorization",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_history_rmse": train_history,
        "val_history_rmse": val_history_rmse,
        "val_history_mae": val_history_mae,
        "train_rows": int(len(train_ratings)),
        "n_users": int(len(user_to_index)),
        "n_items": int(len(item_to_index)),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return model, results, user_to_index, item_to_index


def save_metrics(filename: str, metrics: dict) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / filename
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_mf_artifacts(
    filename: str,
    model: MatrixFactorization,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    np.savez(
        path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        user_bias=model.user_bias,
        item_bias=model.item_bias,
        global_bias=model.global_bias,
        user_ids=np.asarray(list(user_to_index.keys()), dtype=np.int64),
        item_ids=np.asarray(list(item_to_index.keys()), dtype=np.int64),
    )
    return path
