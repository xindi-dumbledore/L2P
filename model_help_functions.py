import numpy as np
from itertools import combinations
import random


def create_pairwise_train_two_pair(X_train, y_train, sort_index):
    """
    Create pairwise feature and label data on training set.

    Keyword arguments:
    X_train -- feature matrix of the data
    y_train -- target variable of the data
    sort_index -- the original index correponsinding to the data

    Output:
    yp -- pairwise relationship
    X_g1 -- first book feature
    X_g2 -- second book feature
    """
    comb = combinations(range(X_train.shape[0]), 2)
    X_g1, X_g2, yp = [], [], []
    pairs = []
    for (i, j) in comb:
        if y_train[i] == y_train[j]:
            continue
        # add pair i,j
        yp.append(np.sign(y_train[i] - y_train[j]))
        X_g1.append(X_train[i])
        X_g2.append(X_train[j])
        pairs.append((sort_index[i], sort_index[j]))
        # add pair j,i
        yp.append(-1 * np.sign(y_train[i] - y_train[j]))  # relative difference
        X_g1.append(X_train[j])
        X_g2.append(X_train[i])
        pairs.append((sort_index[j], sort_index[i]))
    # Xp, yp, diff = map(np.asanyarray, (Xp, yp, diff))
    X_g1, X_g2, yp = map(np.asanyarray, (X_g1, X_g2, yp))
    X_g = np.hstack([X_g1, X_g2])  # concatenate
    return X_g, yp, pairs


def create_pairwise_train_two_pair_efficient(X_train, y_train, sort_index, band, n_sample):
    """
    Create pairwise feature and label data on training set using efficient algorithm.

    Keyword arguments:
    X_train -- feature matrix of the data
    y_train -- target variable of the data
    sort_index -- the original index correponsinding to the data
    band -- the band to define "neighbor"
    n_sample -- number of samples to get for "non-neighbors"

    Output:
    yp -- pairwise relationship
    X_g1 -- first book feature
    X_g2 -- second book feature
    """
    X_g1, X_g2, yp = [], [], []
    pairs = []
    # the y_train are from the highest number to lowest number
    for index in range(len(y_train)):
        # left_neighbors = sort_index[max(0, index - band): index]
        right_neighbors = sort_index[
            (index + 1): min(index + band, len(y_train) - 1)]
        top = sort_index[: max(0, index - band)]
        bottom = sort_index[
            min(index + band, len(y_train) - 1): len(y_train) - 1]
        try:
            sample_top = random.sample(top, n_sample)
        except:
            sample_top = []
        try:
            sample_bottom = random.sample(bottom, n_sample)
        except:
            sample_bottom = []
        for (i, j) in zip([index] * (len(right_neighbors) + len(sample_top) + len(sample_bottom)), list(right_neighbors) + sample_top + sample_bottom):
            if y_train[i] == y_train[j]:
                continue
            # add pair i,j
            yp.append(np.sign(y_train[i] - y_train[j]))
            X_g1.append(X_train[i])
            X_g2.append(X_train[j])
            pairs.append((sort_index[i], sort_index[j]))
            # add pair j,i
            # relative difference
            yp.append(-1 * np.sign(y_train[i] - y_train[j]))
            X_g1.append(X_train[j])
            X_g2.append(X_train[i])
            pairs.append((sort_index[j], sort_index[i]))
    X_g1, X_g2, yp = map(np.asanyarray, (X_g1, X_g2, yp))
    X_g = np.hstack([X_g1, X_g2])  # concatenate
    return X_g, yp, pairs
