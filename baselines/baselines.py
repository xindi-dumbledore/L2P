import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
from Heavy_tail_regression import heavy_tail_regression
from sklearn.metrics import confusion_matrix, precision_score, r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model, neural_network, ensemble, kernel_ridge, neighbors
from xgboost import XGBRegressor
import sys
import json
from collections import Counter
import math
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sns.set_context('poster')
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})


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


def analysis(predictions, binning_threshold, data_name, method_name):
    # save prdictions
    pickle.dump(predictions, open("%s/prediction_%s.pickle" %
                                  (data_name, method_name), "wb"))


def kernel_ridge_reg_fit(X, y, data_name, binning_threshold=0.75, n_splits=5, log=True):
    """
    Build Ridge regression model
    Input: X - regression_X, original
           y - regression_y, original
           data_name - the name of the data
           binning_threshold - threshold used for vertical log binning
           log - whether take the log of the variables for preprocessing
    """
    # Scale the data
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    # Prepare for Stratified K Fold
    # Split y
    bin_result_dict, bin_edge, _ = vertical_log_binning(
        binning_threshold, dict(zip(range(len(list(y))), list(y))))
    split_y = list(bin_result_dict.values())
    # Cross Validation experiment
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = []
    for train, test in cv.split(X_scaled, split_y):
        y_test = y[test].flatten()
        ridge_reg = kernel_ridge.KernelRidge(
            alpha=10**(-5), gamma=5*10**(-5), kernel="rbf")
        ridge_reg.fit(X_scaled[train], y_scaled[train])
        predict_this_fold = list(
            ridge_reg.predict(X_scaled[test]).flatten())
        predict_this_fold_rescale = 10**(
            y_scaler.inverse_transform(predict_this_fold))
        predictions.append((y_test, predict_this_fold_rescale))
    return predictions


def neural_network_reg_fit(X, y, data_name, binning_threshold=0.75, n_splits=5, log=True):
    """
    Build Neural Network model
    Input:
           X - regression_X, original
           y - regression_y, original
           data_name - the name of the data
           binning_threshold - threshold used for vertical log binning
           log - whether take the log of the variables for preprocessing
    """

    # Scale the data
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    # Prepare for Stratified K Fold
    # Split y
    bin_result_dict, bin_edge, _ = vertical_log_binning(
        binning_threshold, dict(zip(range(len(list(y))), list(y))))
    split_y = list(bin_result_dict.values())
    # Cross Validation experiment
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = []
    for train, test in cv.split(X_scaled, split_y):
        y_test = y[test].flatten()
        NN = neural_network.MLPRegressor(
            hidden_layer_sizes=(5, 5, 5), random_state=0)
        NN.fit(X_scaled[train], y_scaled[train])
        predict_this_fold = list(
            NN.predict(X_scaled[test]).flatten())
        predict_this_fold_rescale = 10**(
            y_scaler.inverse_transform(predict_this_fold))
        predictions.append((y_test, predict_this_fold_rescale))
    return predictions


def xgb_fit(X, y, data_name, binning_threshold=0.75, n_splits=5, log=True):
    """
    Build Neural Network model
    Input: 
           X - regression_X, original
           y - regression_y, original
           data_name - the name of the data
           binning_threshold - threshold used for vertical log binning
           log - whether take the log of the variables for preprocessing
    """
    # Scale the data
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    # Prepare for Stratified K Fold
    # Split y
    bin_result_dict, bin_edge, _ = vertical_log_binning(
        binning_threshold, dict(zip(range(len(list(y))), list(y))))
    split_y = list(bin_result_dict.values())
    # Cross Validation experiment
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = []
    for train, test in cv.split(X_scaled, split_y):
        y_test = y[test].flatten()
        xgb = XGBRegressor(n_estimators=300, max_depth = 3, learning_rate = 0.05, random_state=0)
        xgb.fit(X_scaled[train], y_scaled[train])
        predict_this_fold = list(
            xgb.predict(X_scaled[test]).flatten())
        predict_this_fold_rescale = 10**(
            y_scaler.inverse_transform(predict_this_fold))
        predictions.append((y_test, predict_this_fold_rescale))
    return predictions


def KNN_fit(X, y, data_name, binning_threshold=0.75, n_splits=5, log=True):
    """
    Build Neural Network model
    Input: 
           X - regression_X, original
           y - regression_y, original
           data_name - the name of the data
           binning_threshold - threshold used for vertical log binning
           log - whether take the log of the variables for preprocessing
    """
    # Scale the data
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    # Prepare for Stratified K Fold
    # Split y
    bin_result_dict, bin_edge, _ = vertical_log_binning(
        binning_threshold, dict(zip(range(len(list(y))), list(y))))
    split_y = list(bin_result_dict.values())
    # Cross Validation experiment
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = []
    for train, test in cv.split(X_scaled, split_y):
        y_test = y[test].flatten()
        neigh = neighbors.KNeighborsRegressor(n_neighbors=24)
        neigh.fit(X_scaled[train], y_scaled[train])
        predict_this_fold = list(
            neigh.predict(X_scaled[test]).flatten())
        predict_this_fold_rescale = 10**(
            y_scaler.inverse_transform(predict_this_fold))
        predictions.append((y_test, predict_this_fold_rescale))
    return predictions


def random_fit(X, y, data_name, binning_threshold=0.75, n_splits=5, log=True):
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    # Prepare for Stratified K Fold
    # Split y
    bin_result_dict, bin_edge, _ = vertical_log_binning(
        binning_threshold, dict(zip(range(len(list(y))), list(y))))
    split_y = list(bin_result_dict.values())
    # Cross Validation experiment
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = []
    for train, test in cv.split(X_scaled, split_y):
        y_test = y[test].flatten()
        predict = y_test.copy()
        np.random.shuffle(predict)
        predictions.append((y_test, predict))
    return predictions


def heavy_reg_fit(X, y, data_name, binning_threshold=0.75, n_splits=5, log=True):
    # Split y
    bin_result_dict, bin_edge, _ = vertical_log_binning(
        binning_threshold, dict(zip(range(len(list(y))), list(y))))
    split_y = list(bin_result_dict.values())
    # Scale the data
    X_scaled, y_scaled, X_scaler, y_scaler = scale_data(X, y)
    # pack up data
    X_scaled = np.hstack(
        [np.ones(len(X_scaled)).reshape(len(X_scaled), -1), X_scaled])
    data = np.hstack([X_scaled, y_scaled])
    # initialization
    n = len(data)
    delta = 0.05
    k = int(math.ceil(4.5 * np.log(1 / delta)))
    # parameter search on lambda
    lbd_list = [0.01, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
    error_list_all = []
    error_min = float("inf")
    best_params = []
    plt.clf()
    error_list = []
    true = data[:, data.shape[1] - 1]
    for lbd in lbd_list:
        clf, new_theta = heavy_tail_regression(data, k, lbd)
        prediction = np.dot(data[:, 0:data.shape[1] - 1], new_theta)
        error = math.sqrt(np.mean((prediction - true)**2))
        error_list.append(error)
        if error < error_min:
            error_min = error
            best_params = [k, lbd, new_theta]
    # prediction using best_lbd, cross validation
    lbd = best_params[1]
    print(best_params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    predictions = []
    for train, test in cv.split(X_scaled, split_y):
        y_test = y[test].flatten()
        train_data = np.hstack([X_scaled[train], y_scaled[train]])
        clf, new_theta = heavy_tail_regression(train_data, k, lbd)
        predict_this_fold = np.dot(X_scaled[test], new_theta)
        predict_this_fold_rescale = 10**(
            y_scaler.inverse_transform(predict_this_fold))
        predictions.append((y_test, predict_this_fold_rescale))
    return predictions


def main(argv):
    fname, data_name, binning_threshold, n_splits = argv
    binning_threshold = float(binning_threshold)
    n_splits = int(n_splits)
    feature_mapping = {"art": "art_feature", "citation": "citation_feature", "fiction": "simple_book_features", "fiction_complex": "all_book_features",
                       "nonfiction": "simple_book_features",
                       "nonfiction_complex": "all_book_features", "compas": "compas_feature", "white": "compas_feature", "movie": "movie_feature", "synthetic": "synthetic_feature"}
    target_mapping = {"art": "price", "citation": "c5norm", "fiction": "One Year Sale", "fiction_complex": "One Year Sale",
                      "nonfiction": "One Year Sale", "nonfiction_complex": "One Year Sale", "compas": "waiting_time_from_previous", "white": "waiting_time_from_previous", "movie": "gross", "synthetic": "9"}
    feature_fname = feature_mapping[data_name]

    data = pd.read_csv("data_files/%s.csv" % fname)
    print(data.head())
    print(feature_fname)
    feature = json.load(open("data_files/%s.json" % feature_fname))
    target = target_mapping[data_name]
    data = data.fillna(0)
    for each_feature in feature + [target]:
        data[each_feature] = [value if value >
                              0 else 0.0001 for value in data[each_feature]]
    X = data[feature].values
    y = data[target].values.reshape(-1, 1)
    print(len(data))
    # make directory
    try:
        os.mkdir(data_name)
    except:
        pass
    # run experiments
    KLR_result = analysis(kernel_ridge_reg_fit(
        X, y, data_name, binning_threshold, n_splits, log=True), binning_threshold, data_name, "KLR")
    NN_result = analysis(neural_network_reg_fit(
        X, y, data_name, binning_threshold, n_splits=5, log=True), binning_threshold, data_name, "NN")
    kNN_result = analysis(
        KNN_fit(X, y, data_name, binning_threshold, n_splits, log=True), binning_threshold, data_name, "kNN")
    HLR_result = analysis(heavy_reg_fit(
        X, y, data_name, binning_threshold, n_splits, log=True), binning_threshold, data_name, "HLR")
    XGB_result = analysis(
        xgb_fit(X, y, data_name, binning_threshold=0.75, n_splits=5, log=True), binning_threshold, data_name, "XGB")
    random_result = analysis(random_fit(X, y, data_name, binning_threshold=0.75,
                                        n_splits=5, log=True), binning_threshold, data_name, "random")
    LtP_prediction = pickle.load(
        open("../LtP_results/%s/prediction_LtP.pickle" % data_name, "rb"))
    LtP_result = analysis(LtP_prediction, binning_threshold, data_name, "LtP")
    # score_dict = {}
    # score_dict["LR"] = {"R2": LR_result[0],
    #                     "High-end RMSE": LR_result[1], "RMSE": LR_result[2]}
    # score_dict["KLR"] = {"R2": KLR_result[0],
    #                      "High-end RMSE": KLR_result[1], "RMSE": KLR_result[2]}
    # score_dict["NN"] = {"R2": NN_result[0],
    #                     "High-end RMSE": NN_result[1], "RMSE": NN_result[2]}
    # score_dict["RF"] = {"R2": RF_result[0],
    #                     "High-end RMSE": RF_result[1], "RMSE": RF_result[2]}
    # score_dict["kNN"] = {"R2": kNN_result[0],
    #                      "High-end RMSE": kNN_result[1], "RMSE": kNN_result[2]}
    # score_dict["HLR"] = {"R2": HLR_result[0],
    #                      "High-end RMSE": HLR_result[1], "RMSE": HLR_result[2]}
    # score_dict["XGB"] = {"R2": XGB_result[0],
    #                      "High-end RMSE": XGB_result[1], "RMSE": XGB_result[2]}
    # score_dict["LtP"] = {"R2": LtP_result[0],
    #                      "High-end RMSE": LtP_result[1], "RMSE": LtP_result[2]}
    # json.dump(score_dict, open("%s/score.json" % data_name, "w"), indent=4)

    # plot Q-Q on top of each other


if __name__ == "__main__":
    main(sys.argv[1:])
