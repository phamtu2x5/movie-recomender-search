import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        features = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(features).squeeze(-1)
