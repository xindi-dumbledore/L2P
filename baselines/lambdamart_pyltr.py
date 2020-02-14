"""
    Data Format:
    [score, query_id, features]

"""
import pyltr
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from collections import Counter
import json
import scipy
import matplotlib.pyplot as plt
import pickle
import sys


def vertical_log_binning(p, data):
    """Create vertical log_binning. Used for peak sale."""
    import math
    import operator
    index, value = zip(*sorted(data.items(), key=operator.itemgetter(1)))
    bin_result = []
    value = list(value)
    bin_edge = [min(value)]
    i = 1
    while len(value) > 0:
        num_to_bin = int(math.ceil(p * len(value)))
        # print num_to_bin
        edge_value = value[num_to_bin - 1]
        bin_edge.append(edge_value)
        to_bin = list(filter(lambda x: x <= edge_value, value))
        bin_result += [i] * len(to_bin)
        value = list(filter(lambda x: x > edge_value, value))
        # print len(bin_result) + len(value)
        i += 1
        # print '\n'
    bin_result_dict = dict(zip(index, bin_result))
    bin_distri = Counter(bin_result_dict.values())
    # print len(index), len(bin_result)
    return bin_result_dict, bin_edge, bin_distri


def scale_data(X, y):
    """Scale data to have 0 mean and 1 std"""
    from sklearn import preprocessing
    # Log the variables
    X = np.log10(X)
    y = np.log10(y)
    X_scaler = preprocessing.StandardScaler().fit(X)
    y_scaler = preprocessing.StandardScaler().fit(y)
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y)
    return X_scaled, y_scaled, X_scaler, y_scaler


def read_data(fname, data_name):
    data = pd.read_csv("data_files/%s.csv" % fname)
    feature_fname = FEATURE_MAPPING[data_name]
    print(feature_fname)
    feature = json.load(open("data_files/%s.json" % feature_fname))
    target = TARGET_MAPPING[data_name]
    data = data.fillna(0)
    for each_feature in feature + [target]:
        data[each_feature] = [value if value >
                              0 else 0.0001 for value in data[each_feature]]
    X = data[feature].values
    y = data[target].values.reshape(-1, 1)
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    query_id = np.array([1] * len(y)).reshape(-1, 1)
    data = np.hstack((y_scaled, query_id, X_scaled))
    return data, y, y_scaler


def cross_validation(data, binning_threshold, y, y_scaler, n_estimators, max_depth, n_splits=5):
    bin_result_dict, bin_edge, _ = vertical_log_binning(
        binning_threshold, dict(zip(range(len(list(y))), list(y))))
    split_y = list(bin_result_dict.values())
    # Cross Validation experiment
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = []
    for train, test in cv.split(data, split_y):
        print(len(train), len(test))
        train_data = data[train]
        test_data = data[test]
        TX, Ty, Tqids = train_data[:, 2:], train_data[:, 0], train_data[:, 1]
        EX, Ey, Eqids = test_data[:, 2:], test_data[:, 0], test_data[:, 1]
        metric = pyltr.metrics.roc.AUCROC()
        monitor = pyltr.models.monitors.ValidationMonitor(
            EX, Ey, Eqids, metric=metric, stop_after=50)
        model = pyltr.models.LambdaMART(
            metric=metric,
            # subsample=0.5,
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_features="log2",
            query_subsample=1,
            max_depth=max_depth,
            min_samples_leaf=64,
            verbose=1,
            random_state=1
        )
        model.fit(TX, Ty, Tqids, monitor=monitor)
        predict_this_fold = []
        for each_test in test_data:
            temp = np.vstack((train_data, each_test))
            tempX, tempy, tempqids = temp[:, 2:], temp[:, 0], temp[:, 1]
            temppred = model.predict(tempX)
            argsorted = list(np.argsort(temppred))
            target_position = argsorted.index(len(temppred) - 1)
            adjacentrank = [max(0, target_position - 1),
                            min(target_position + 1, len(temppred) - 1)]
            adjacentsale = 10**(y_scaler.inverse_transform(
                tempy[adjacentrank].reshape(-1, 1)))
            predictsale = np.mean(adjacentsale)
            predict_this_fold.append(predictsale)
        true = 10**(y_scaler.inverse_transform(test_data[:, 0]))
        # plt.loglog(predict_this_fold, true, 'o')
        # plt.show()
        predictions.append((true, predict_this_fold))
    return predictions

FEATURE_MAPPING = {"art": "art_feature", "citation": "citation_feature", "fiction": "simple_book_features", "fiction_complex": "all_book_features",
                   "nonfiction": "simple_book_features",
                   "nonfiction_complex": "all_book_features", "compas": "compas_feature", "white": "compas_feature", "movie": "movie_feature", "synthetic": "synthetic_feature"}
TARGET_MAPPING = {"art": "price", "citation": "c5norm", "fiction": "One Year Sale", "fiction_complex": "One Year Sale", "nonfiction": "One Year Sale",
                  "nonfiction_complex": "One Year Sale", "compas": "waiting_time_from_previous", "white": "waiting_time_from_previous", "movie": "gross", "synthetic": "9"}


def main(argv):
    fname, data_name, binning_threshold, n_estimators, max_depth, n_splits = argv
    binning_threshold = float(binning_threshold)
    n_splits = int(n_splits)
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    # binning_threshold = 0.75
    # nsplit = 5
    data, y, y_scaler = read_data(fname, data_name)
    predictions = cross_validation(
        data, binning_threshold, y, y_scaler, n_estimators, max_depth, n_splits=n_splits)
    pickle.dump(predictions, open("%s/prediction_%s.pickle" %
                                  (data_name.split('_')[0], "lambdamart"), "wb"))

if __name__ == '__main__':
    main(sys.argv[1:])

# python lambdamart_pyltr.py fiction fiction_complex 0.75 500 5 5
# python lambdamart_pyltr.py nonfiction nonfiction_complex 0.75 1000 5 5
# python lambdamart_pyltr.py art_painting_LtP_sample art 0.94 500 5 5

# binning_threshold = 0.94
# nsplit = 5
# data, y, y_scaler = read_data("art_painting_LtP_sample", "art")
# predictions = cross_validation(data, binning_threshold, y, y_scaler, n_splits=5)
# pickle.dump(predictions, open("%s/prediction_%s.pickle" %
#                               ("art", "lambdamart"), "wb"))
