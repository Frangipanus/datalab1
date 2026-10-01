import numpy as np


def global_mean(train):
    mean = np.nanmean(train)
    return np.full(train.shape, mean)


def movie_mean(train):
    global_avg = np.nanmean(train)
    means = np.nanmean(train, axis=0)
    means = np.where(np.isnan(means), global_avg, means)
    return np.tile(means, (train.shape[0], 1))


def user_mean(train):
    global_avg = np.nanmean(train)
    means = np.nanmean(train, axis=1)
    means = np.where(np.isnan(means), global_avg, means)
    return np.repeat(means[:, np.newaxis], train.shape[1], axis=1)