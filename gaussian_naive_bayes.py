import numpy as np
import pandas as pd


def gaussian_function(x, mean, std):
    exponent = np.exp(-((x-mean) ** 2 / (2 * std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


def get_statistic(X: np.ndarray):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    mean_std_arr = np.vstack([mean, std]).T
    mean_std_df = pd.DataFrame(mean_std_arr, columns=['mean', 'std'])
    return mean_std_df


def get_labels_statistic(X: np.ndarray, y: np.ndarray):
    labels_statistics = {}
    labels = np.unique(y)
    for label in labels:
        label_X = X[y == label]
        std_and_mean_df = get_statistic(label_X)
        labels_statistics[label] = std_and_mean_df
    return labels_statistics


def predict_likelihood_for_1d(X: np.array, labels_statistic: dict):
    """
    :param X: one dimensional array
    :param labels_statistic:
    :return: likelihood
    """
    if is_array_1d(X):
        # if we have one or more 1-length dimension
        #  for example: X.shape=(1, 1, 3) or X.shape=(3, 1)
        X = np.ravel(X)
    else:
        raise ValueError('X should be 1D')

    # check features equality
    if X.shape[0] != get_n_features_of_labels_statistic(labels_statistic):
        raise ValueError('X should have equal number of features to statistic')

    likelihood = {}
    for label, label_statistic in labels_statistic.items():
        label_likelihood = 1
        for feature_ind in range(X.size):
            mean = label_statistic.loc[feature_ind, 'mean']
            std = label_statistic.loc[feature_ind, 'std']
            gaussian_prob = gaussian_function(X[feature_ind], mean, std)
            label_likelihood *= gaussian_prob
        likelihood[label] = label_likelihood
    return likelihood


def is_array_1d(arr: np.array):
    return np.prod(arr.shape) == np.max(arr.shape)


def get_n_features_of_labels_statistic(labels_statistic: dict):
    first_value = list(labels_statistic.items())[0][1]
    return first_value.shape[0]


def predict_likelihood(X: np.array, labels_statistic: dict):
    if len(X.shape) == 1:
        # create column-matrix from
        X = X.reshape(1, -1)
    likelihood_dfs_list = []
    for x_row in X:
        likelihood = predict_likelihood_for_1d(x_row, labels_statistic)
        # index key argument don't matter, it is just created to avoid error
        likelihood_df = pd.DataFrame(likelihood, index=[0])
        likelihood_dfs_list.append(likelihood_df)
    # return dataframe with columns: labels, rows: X rows
    return pd.concat(likelihood_dfs_list, ignore_index=True)


def predict_labels_from_likelihoods(likelihoods):
    return likelihoods.idxmax(axis=1)


def predict(X, labels_statistic):
    likelyhoods = predict_likelihood(X, labels_statistic)
    return predict_labels_from_likelihoods(likelyhoods)


