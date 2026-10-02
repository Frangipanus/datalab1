import torch
import numpy as np


def als_with_biases(train, rank=10, reg=0.1, n_iter=10):
    R = torch.tensor(train, dtype=torch.float32).T # R est [n_movies, n_users]
    n_movies, n_users = R.shape
    
    torch.manual_seed(0) 
    
    # Initialisation des facteurs latents et des biais
    I = torch.randn(n_movies, rank)
    U = torch.randn(n_users, rank)
    b_i = torch.zeros(n_movies)
    b_u = torch.zeros(n_users)
    
    # Calcul de la moyenne globale
    mask_global = ~torch.isnan(R)
    mu = R[mask_global].mean()
    
    # Matrice de régularisation de taille (rank + 1) pour inclure le biais.
    # En général, on ne régularise pas (ou très peu) les termes de biais, 
    # donc on met la première valeur de la diagonale à 0.
    reg_mat = reg * torch.eye(rank + 1)
    reg_mat[0, 0] = 0.0 

    for iteration in range(n_iter):

        # 1. Fixer les items (I, b_i), optimiser les utilisateurs (U, b_u)
        for u in range(n_users):
            ratings_u = R[:, u]
            mask = ~torch.isnan(ratings_u)
            
            if not mask.any():
                continue # Sécurité si un utilisateur n'a aucune note

            I_u = I[mask]
            b_i_u = b_i[mask]
            ratings = ratings_u[mask]

            # La cible devient la note moins la moyenne globale et le biais du film
            target = ratings - mu - b_i_u

            # On ajoute une colonne de 1 pour apprendre b_u en même temps que U[u]
            ones = torch.ones(I_u.shape[0], 1)
            I_u_aug = torch.cat([ones, I_u], dim=1)

            A = I_u_aug.T @ I_u_aug + reg_mat
            b = I_u_aug.T @ target

            res = torch.linalg.solve(A, b)
            b_u[u] = res[0]   # Le premier élément est le biais utilisateur
            U[u] = res[1:]    # Le reste correspond aux facteurs latents

        # 2. Fixer les utilisateurs (U, b_u), optimiser les items (I, b_i)
        for i in range(n_movies):
            users_r = R[i, :]
            mask = ~torch.isnan(users_r)
            
            if not mask.any():
                continue

            U_r = U[mask]
            b_u_r = b_u[mask]
            users = users_r[mask]

            # La cible devient la note moins la moyenne globale et le biais utilisateur
            target = users - mu - b_u_r

            # On ajoute une colonne de 1 pour apprendre b_i en même temps que I[i]
            ones = torch.ones(U_r.shape[0], 1)
            U_r_aug = torch.cat([ones, U_r], dim=1)

            A = U_r_aug.T @ U_r_aug + reg_mat
            b = U_r_aug.T @ target

            res = torch.linalg.solve(A, b)
            b_i[i] = res[0]   # Le premier élément est le biais du film
            I[i] = res[1:]    # Le reste correspond aux facteurs latents

    # Reconstitution de la matrice de prédiction : mu + b_i + b_u + I @ U.T
    # On utilise unsqueeze pour diffuser (broadcast) les vecteurs de biais correctement
    predictions = mu + b_i.unsqueeze(1) + b_u.unsqueeze(0) + (I @ U.T)
    
    # On transpose à nouveau pour revenir à la forme d'entrée [n_users, n_movies]
    return predictions.T.numpy()
