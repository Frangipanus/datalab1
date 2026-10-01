import numpy as np

from models.als_paul import als


def rmse(target, predictions):
    mask = ~np.isnan(target)
    return np.sqrt(np.mean((target[mask] - predictions[mask]) ** 2))


train = np.load("data/ratings_train.npy")
test = np.load("data/ratings_test.npy")

ranks = [5, 10, 20, 40, 80,100,125,150,175,200]
regs = [0.001, 0.01, 0.1, 1, 10, 100]
n_iters = [2, 5, 10, 20, 50, 70,100]

best_score = float("inf")
best_params = None
i = 0
maxi = len(ranks)*len(regs)*len(n_iters)
for rank in ranks:
    for reg in regs:
        for n_iter in n_iters:
            i+=1
            predictions = als(
                train,
                rank=rank,
                reg=reg,
                n_iter=n_iter
            )

            train_score = rmse(train, predictions)
            test_score = rmse(test, predictions)

            print(
                f"{i} out of {maxi} "
                f"rank={rank:<3} reg={reg:<6} "
                f"train={train_score:.4f} test={test_score:.4f}"
            )

            if test_score < best_score:
                best_score = test_score
                best_params = (rank, reg, n_iter)

print()
print("Best configuration:")
print("rank =", best_params[0])
print("reg =", best_params[1])
print("epoch =", best_params[2])
print("test RMSE =", best_score)