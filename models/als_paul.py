import torch
import numpy as np 


import torch


def als(train, rank=10, reg=0.1, n_iter=10):
    R = torch.tensor(train, dtype=torch.float32).T #on garde la convention des slides que l'entrée c'est film x usert
    n_movies, n_users = R.shape
    torch.manual_seed(0) #pour tester les hyperparametres
    I = torch.randn(n_movies, rank)
    U = torch.randn(n_users, rank)
    for iteration in range(n_iter):

        # I est fixé
        for u in range(n_users):
            ratings_u = R[:, u]
            mask = ~torch.isnan(ratings_u)

            I_u = I[mask]
            ratings = ratings_u[mask]

            A = I_u.T @ I_u + reg * torch.eye(rank)
            b = I_u.T @ ratings

            U[u] = torch.linalg.solve(A, b)

        # U est fixé
        for i in range(n_movies):
            users_r = R[i,:]
            mask = ~torch.isnan(users_r)

            U_r = U[mask]
            users = users_r[mask]

            A = U_r.T @ U_r + reg*torch.eye(rank)
            b = U_r.T @ users

            I[i] = torch.linalg.solve(A, b)
    predictions = (I @ U.T).T
    return predictions.numpy()