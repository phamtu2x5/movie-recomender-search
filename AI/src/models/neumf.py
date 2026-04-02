import torch
import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()

        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(embedding_dim + 32, 1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_mf = self.user_embedding_mf(user_idx)
        item_mf = self.item_embedding_mf(item_idx)
        user_mlp = self.user_embedding_mlp(user_idx)
        item_mlp = self.item_embedding_mlp(item_idx)

        mf_part = user_mf * item_mf
        mlp_part = self.mlp(torch.cat([user_mlp, item_mlp], dim=-1))
        final_features = torch.cat([mf_part, mlp_part], dim=-1)
        return self.output(final_features).squeeze(-1)
