import numpy as np

from models.baseline import global_mean, movie_mean, user_mean


def rmse(test, predictions):
    mask = ~np.isnan(test)
    return np.sqrt(np.mean((test[mask] - predictions[mask]) ** 2))


train = np.load("data/ratings_train.npy")
test = np.load("data/ratings_test.npy")

models = {
    "Global mean": global_mean,
    "Movie mean": movie_mean,
    "User mean": user_mean,
}

for name, model in models.items():
    predictions = model(train)
    score = rmse(test, predictions)
    print(f"{name}: {score:.4f}")