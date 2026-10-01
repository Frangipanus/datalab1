import numpy as np

from models.baseline import global_mean, movie_mean, user_mean
from models.als_paul import als

def rmse(test, predictions):
    mask = ~np.isnan(test)
    return np.sqrt(np.mean((test[mask] - predictions[mask]) ** 2))


train = np.load("data/ratings_train.npy")
test = np.load("data/ratings_test.npy")
print(train.shape)
models = {
    "Global mean": global_mean,
    "Movie mean": movie_mean,
    "User mean": user_mean,
    "ALS": als,
}

results = []

for name, model in models.items():
    predictions = model(train)
    score = rmse(test, predictions)
    results.append((name, score))


print()
print("+----------------------+----------+")
print("| Method               | RMSE     |")
print("+----------------------+----------+")

for name, score in results:
    print(f"| {name:<20} | {score:>8.4f} |")

print("+----------------------+----------+")