import argparse
import numpy as np

from models.baseline import global_mean, movie_mean, user_mean
from models.als_paul import als


def rmse(test, predictions):
    mask = ~np.isnan(test)
    return np.sqrt(np.mean((test[mask] - predictions[mask]) ** 2))


def main(train_path, test_path):
    train = np.load(train_path)
    test = np.load(test_path)

    print(f"Train shape: {train.shape}, {np.sum(~np.isnan(train))} ratings")
    print(f"Test shape: {test.shape}, {np.sum(~np.isnan(test))} ratings")

    models = {
        "Global mean": global_mean,
        "Movie mean": movie_mean,
        "User mean": user_mean,
        "ALS": als,
    }

    results = []

    for name, model in models.items():
        try:
            predictions = model(train)
        except:
            results.append((name, np.nan))
            continue
        score = rmse(test, predictions)
        results.append((name, score))

    print()
    print("+----------------------+----------+")
    print("| Method               | RMSE     |")
    print("+----------------------+----------+")

    for name, score in results:
        print(f"| {name:<20} | {score:>8.4f} |")

    print("+----------------------+----------+")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate models.")
    parser.add_argument(
        "--train",
        type=str,
        default="data/ratings_train.npy",
        help="Path to the training ratings file (NumPy .npy format)."
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/ratings_test.npy",
        help="Path to the testing ratings file (NumPy .npy format)."
    )
    args = parser.parse_args()

    main(args.train, args.test)
