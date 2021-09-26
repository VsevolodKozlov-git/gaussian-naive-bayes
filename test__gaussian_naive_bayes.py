import unittest
import numpy as np
import pandas as pd

from gaussian_naive_bayes import *


class MyTestCase(unittest.TestCase):
    # training data of labels 0
    X0 = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    # training data of labels 1
    X1 = np.array([
        [7, 8, 9],
        [10, 11, 12]
    ])
    # merged X0 and X1
    X_merged = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    # corresponding labels for X_merged
    y = np.array([0, 0, 1, 1])

    def test__get_label_statistic(self):
        statistic = get_statistic(self.X0)
        self.assertEqual(statistic.shape, (3, 2))

        std_equal_mask = self.X0.std(axis=0) == statistic['std']
        self.assertTrue(np.all(std_equal_mask))

        mean_equal_mask = self.X0.mean(axis=0) == statistic['mean']
        self.assertTrue(np.all(mean_equal_mask))

    def test__labels_statistic(self):
        y = np.array([0, 0, 1, 1])
        statistic = get_labels_statistic(self.X_merged, y)
        self.assertTrue(statistic[0].equals(get_statistic(self.X0)))
        self.assertTrue(statistic[1].equals(get_statistic(self.X1)))

    def test_likelihood_for_1d_normal_use(self):
        x_to_predict = np.array([0, 1, 2])
        # creating statistic manually
        std_arr_0 = np.std(self.X0, axis = 0)
        mean_arr_0 = np.mean(self.X0, axis=0)
        # counting likelihood manually
        expected_likelyhood_0 = 1
        for x,  mean, std in zip(x_to_predict, mean_arr_0, std_arr_0):
            expected_likelyhood_0 *= gaussian_function(x, mean, std)

        statistic = get_labels_statistic(self.X_merged, self.y)
        predicted_likelyhood = predict_likelihood_for_1d(x_to_predict, statistic)

        self.assertEqual(expected_likelyhood_0, predicted_likelyhood[0])

    def test_likelihood_for_1d_with_another_labels(self):
        statistics_of_num_labels = get_labels_statistic(self.X_merged,
                                                        self.y)
        y_letter_labels = np.array(['a', 'a', 'b', 'b'])
        statistics_of_letter_labels = get_labels_statistic(self.X_merged,
                                                           y_letter_labels)
        x = np.array([1, 2, 3])
        likelihood_num = predict_likelihood_for_1d(x, statistics_of_num_labels)
        likelihood_letter = predict_likelihood_for_1d(x, statistics_of_letter_labels)

        self.assertEqual(likelihood_num[0], likelihood_letter['a'])
        self.assertEqual(likelihood_num[1], likelihood_letter['b'])

    def test_likelihood_for_1d_with_row_vector(self):
        self.X_merged = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        self.y = np.array([0, 0, 1, 1])
        statistic = get_labels_statistic(self.X_merged, self.y)

        # try to create emergency situation
        x = np.array([0, 1, 2])[:, np.newaxis]
        predict_likelihood_for_1d(x, statistic)

        x = x.reshape(1, 1, 3)
        predict_likelihood_for_1d(x, statistic)

    def test_likelihood_for_1d_with_2d_matrix(self):
        x = np.array([
            [1, 2, 3],
            [3, 4, 5]
        ])
        statistics = get_labels_statistic(self.X_merged, self.y)
        try:
            predict_likelihood_for_1d(x, statistics)
            self.assertTrue(False, 'predict_likelihood_for_1d should raise error')
        except ValueError:
            self.assertTrue(True)

    def test_likelihood_for_1d_with_incorrect_feature_vector(self):
        statistics = get_labels_statistic(self.X_merged, self.y)
        x = np.array([3, 4, 5, 6])
        try:
            predict_likelihood_for_1d(x, statistics)
            self.assertTrue(False, 'predict_likelihood_for_1d should raise error')
        except ValueError:
            self.assertTrue(True)

    def test_get_n_features_of_statistic(self):
        stat = get_labels_statistic(self.X_merged, self.y)
        n_features = get_n_features_of_labels_statistic(stat)
        self.assertEqual(3, n_features)

        stat = {'f': np.array([
            [3, 4, 5, 6],
            [7, 8, 9, 10]
        ])}
        n_features = get_n_features_of_labels_statistic(stat)
        self.assertEqual(2, n_features)

    def test_predict_likelihood(self):
        """
        test normal use cases with just splitting array
         and counting all manually
        """
        statistic = get_labels_statistic(self.X_merged, self.y)
        X_1d_0 = np.array([1, 2, 3])
        X_1d_1 = np.array([3, 4, 5])

        X_2d = np.vstack([X_1d_0, X_1d_1])

        likelihood_2d = predict_likelihood(X_2d, statistic)

        likelihood_1d_0 = predict_likelihood_for_1d(X_1d_0, statistic)
        likelihood_1d_1 = predict_likelihood_for_1d(X_1d_1, statistic)
        likelihood_merged = pd.DataFrame([likelihood_1d_0,
                                          likelihood_1d_1])
        self.assertTrue(likelihood_2d.equals(likelihood_merged))

    def test_predict_likelihood__with_1d_case(self):
        statistic = get_labels_statistic(self.X_merged, self.y)
        X_1d = np.array([1, 2, 3])
        pred_of_1d_func = predict_likelihood_for_1d(X_1d, statistic)
        df_1d_func = pd.DataFrame(pred_of_1d_func, index=[0])
        pred_of_general_func = predict_likelihood(X_1d, statistic)
        vals_equals = pred_of_general_func.equals(df_1d_func)
        self.assertTrue(vals_equals)

    def test_predict_labels_from_likelihoods(self):
        likelihoods = pd.DataFrame([
              [0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]
        ], columns=['a', 'b', 'c'])

        labels = predict_labels_from_likelihoods(likelihoods)
        expected_labels = pd.Series(['b', 'a', 'c'])
        self.assertTrue(labels.equals(expected_labels))

    def test_predict(self):
        # choose values of x0 and x1 to have labels 0 and corresponding
        statistic = get_labels_statistic(self.X_merged, self.y)
        x0 = statistic[0].loc[:, 'mean']
        x1 = statistic[1].loc[:, 'mean']
        X = np.vstack([x0, x1])

        predicted = predict(X, statistic)
        exprected_predictions = pd.Series([0, 1])
        self.assertTrue(exprected_predictions.equals(predicted))
    # todo test that predictions on titanic dataset don't cause any errors
    def test_predict_with_titanic(self):
        pass
if __name__ == '__main__':
    unittest.main()
