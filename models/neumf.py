import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=8):
        super().__init__()
        # GMF Embeddings
        self.user_emb_gmf = nn.Embedding(n_users, emb_dim)
        self.item_emb_gmf = nn.Embedding(n_items, emb_dim)
        
        # MLP Embeddings
        self.user_emb_mlp = nn.Embedding(n_users, emb_dim)
        self.item_emb_mlp = nn.Embedding(n_items, emb_dim)
        
        # MLP Layers
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Final prediction layer combining both branches
        self.prediction = nn.Linear(emb_dim + 8, 1)

    def forward(self, user, item):
        # --- Generalized Matrix Factorization (GMF) ---
        gmf_u = self.user_emb_gmf(user)
        gmf_i = self.item_emb_gmf(item)
        gmf_out = gmf_u * gmf_i  # Element-wise product

        # --- Multi-Layer Perceptron (MLP) ---
        mlp_u = self.user_emb_mlp(user)
        mlp_i = self.item_emb_mlp(item)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # --- Concatenation ---
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.prediction(concat)
        
        return out.squeeze()


def complete_matrix_neumf(matrix, epochs=150, lr=0.01):
    n_users, n_items = matrix.shape

    # 1. Prepare training data (Extract indices and values where rating is not NaN)
    users, items = np.where(~np.isnan(matrix))
    ratings = matrix[users, items]

    users_t = torch.tensor(users, dtype=torch.long)
    items_t = torch.tensor(items, dtype=torch.long)
    ratings_t = torch.tensor(ratings, dtype=torch.float32)

    # 2. Initialize Model, Loss (MSE for 0-5 ratings), and Optimizer
    model = NeuMF(n_users, n_items)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3. Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(users_t, items_t)
        loss = criterion(preds, ratings_t)
        loss.backward()
        optimizer.step()

    # 4. Generate complete output matrix
    # Create tensors for every possible user-item pair
    all_users = torch.arange(n_users).repeat_interleave(n_items)
    all_items = torch.arange(n_items).repeat(n_users)

    model.eval()
    with torch.no_grad():
        all_preds = model(all_users, all_items).numpy()

    # Reshape back to (n_users, n_items) and clip to valid rating range (0 to 5)
    complete_matrix = all_preds.reshape(n_users, n_items)
    complete_matrix = np.clip(complete_matrix, 0.0, 5.0)
    
    return complete_matrix
