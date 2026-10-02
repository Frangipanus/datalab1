import argparse
import numpy as np
import time

from models.baseline import global_mean, movie_mean, user_mean
from models.als_paul import als
from models.als_with_biases import als_with_biases
from models.neumf import complete_matrix_neumf as neumf


def rmse(test, predictions):
    mask = ~np.isnan(test)
    return np.sqrt(np.mean((test[mask] - predictions[mask]) ** 2))


def evaluate(train, test, model, atol=0.25):
    start_time = time.time()

    try:
        predictions = model(train)
    except:
        return np.nan, np.nan, np.nan

    elapsed_time = time.time() - start_time

    score = rmse(test, predictions)
    accuracy = np.mean(np.isclose(test[~np.isnan(test)], predictions[~np.isnan(test)], atol=atol)) * 100

    return score, accuracy, elapsed_time


def main(train_path, test_path):
    train = np.load(train_path)
    test = np.load(test_path)

    print(f"Train shape: {train.shape}, {np.sum(~np.isnan(train))} ratings")
    print(f"Test shape: {test.shape}, {np.sum(~np.isnan(test))} ratings")

    models = {
        "Global mean": global_mean,
        "Movie mean": movie_mean,
        "User mean": user_mean,
        "NeuMF": neumf,
        "ALS": als,
        "ALS with biases": als_with_biases
    }

    print()
    print("+----------------------+----------+--------------+----------+")
    print("| Method               | RMSE     | Accuracy (%) | Time (s) |")
    print("+----------------------+----------+--------------+----------+")

    for name, model in models.items():
        score, accuracy, elapsed_time = evaluate(train, test, model)
        print(f"| {name:<20} | {score:>8.4f} | {accuracy:>12.2f} | {elapsed_time:>8.2f} |")

    print("+----------------------+----------+--------------+----------+")


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
