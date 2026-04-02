import json
import copy
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.config import METRICS_DIR, MODELS_DIR
from AI.src.evaluation.metrics import mae, rmse
from AI.src.features.encode import apply_id_mappings, build_id_mappings
from AI.src.models.neumf import NeuMF


@dataclass
class NCFConfig:
    embedding_dim: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 5
    grad_clip_norm: float | None = 5.0
    show_progress: bool = True


def _make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(frame["user_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(frame["item_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(frame["rating"].to_numpy(), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_neumf(
    model: NeuMF,
    frame: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    batch_size: int,
    device: str,
) -> dict:
    mapped = apply_id_mappings(frame, user_to_index, item_to_index)
    if mapped.empty:
        return {"rmse": None, "mae": None, "n_rows": 0}

    loader = _make_loader(mapped, batch_size=batch_size, shuffle=False)
    preds_list = []
    model.eval()
    with torch.no_grad():
        for user_idx, item_idx, _ in loader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            preds = torch.sigmoid(model(user_idx, item_idx)) * 5.0
            preds_list.append(preds.cpu())

    all_preds = torch.cat(preds_list).numpy()
    return {
        "rmse": rmse(mapped["rating"].to_numpy(), all_preds),
        "mae": mae(mapped["rating"].to_numpy(), all_preds),
        "n_rows": int(len(mapped)),
    }


def train_neumf(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame | None = None,
    config: NCFConfig | None = None,
    device: str = "cpu",
) -> tuple[NeuMF, dict, dict[int, int], dict[int, int]]:
    config = config or NCFConfig()
    user_to_index, item_to_index = build_id_mappings(train_ratings)

    train = apply_id_mappings(train_ratings, user_to_index, item_to_index)

    model = NeuMF(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()

    train_loader = _make_loader(train, config.batch_size, shuffle=True)
    train_history_loss: list[float] = []
    val_history_rmse: list[float] = []
    val_history_mae: list[float] = []
    best_val_rmse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    epoch_iterator = range(config.epochs)
    if config.show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc="NeuMF training", leave=False)

    for epoch_idx in epoch_iterator:
        model.train()
        total_loss = 0.0
        batch_iterator = train_loader
        if config.show_progress:
            batch_iterator = tqdm(
                train_loader,
                desc=f"NeuMF Epoch {epoch_idx + 1}/{config.epochs}",
                leave=False,
            )

        for user_idx, item_idx, rating in batch_iterator:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device)

            optimizer.zero_grad()
            preds = torch.sigmoid(model(user_idx, item_idx)) * 5.0
            loss = criterion(preds, rating)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(train_loader), 1)
        val_metrics_epoch = _evaluate_neumf(
            model,
            val_ratings,
            user_to_index,
            item_to_index,
            batch_size=config.batch_size,
            device=device,
        )

        train_history_loss.append(epoch_loss)
        val_history_rmse.append(val_metrics_epoch["rmse"])
        val_history_mae.append(val_metrics_epoch["mae"])

        if val_metrics_epoch["rmse"] is not None and val_metrics_epoch["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics_epoch["rmse"]
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())

        if config.show_progress and hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                train_loss=round(epoch_loss, 4),
                val_rmse=round(val_metrics_epoch["rmse"], 4),
            )

        print(
            f"Epoch {epoch_idx + 1:02d}/{config.epochs:02d} | "
            f"train_loss={epoch_loss:.4f} | "
            f"val_rmse={val_metrics_epoch['rmse']:.4f} | "
            f"val_mae={val_metrics_epoch['mae']:.4f}"
        )

    model.load_state_dict(best_state)

    val_metrics = _evaluate_neumf(
        model,
        val_ratings,
        user_to_index,
        item_to_index,
        batch_size=config.batch_size,
        device=device,
    )
    test_metrics = (
        _evaluate_neumf(
            model,
            test_ratings,
            user_to_index,
            item_to_index,
            batch_size=config.batch_size,
            device=device,
        )
        if test_ratings is not None
        else None
    )
    results = {
        "model": "neumf",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_history_loss": train_history_loss,
        "val_history_rmse": val_history_rmse,
        "val_history_mae": val_history_mae,
        "train_rows": int(len(train_ratings)),
        "n_users": int(len(user_to_index)),
        "n_items": int(len(item_to_index)),
        "device": device,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return model, results, user_to_index, item_to_index


def save_metrics(filename: str, metrics: dict) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / filename
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_neumf_artifacts(
    filename: str,
    model: NeuMF,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
        },
        path,
    )
    return path
