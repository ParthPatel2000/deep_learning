import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class MyKNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

    def predict(self, test_features):
        predictions = []
        for test_sample in test_features:
            distances = np.sqrt(np.sum((self.train_features - test_sample) ** 2, axis=1))
            nearest_neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbor_labels = self.train_labels[nearest_neighbor_indices]

            '''using bincount to count the occurrences of the labels in the array
            and storing them in an array of counts where each index has the
            number of occurrence of that index(number) in the labels array stored in it.
            then using argmax to find the index of the maximum number from the predicted_label.'''
            predicted_label = np.bincount(nearest_neighbor_labels).argmax()
            predictions.append(predicted_label)
        return np.array(predictions)

class MyCV:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = {}

    def fit_one(self, param_dict, train_features, train_labels):
        self.estimator.n_neighbors = param_dict['n_neighbors']
        self.estimator.fit(train_features, train_labels)

    def fit(self, train_features, train_labels):
        kf = KFold(n_splits=self.cv,shuffle=True)
        validation_df_list = []

        for validation_fold, (train_indices, val_indices) \
                in enumerate(kf.split(train_features)):
            subtrain_data = {
                "subtrain": (train_features[train_indices], train_labels[train_indices]),
                "validation_fold": validation_fold
            }
 
            for param_dict in self.param_grid:
                self.fit_one(param_dict, **subtrain_data)
                y_pred = self.estimator.predict(train_features[val_indices])
                accuracy = accuracy_score(train_features[val_indices], y_pred)
                validation_row = {
                    "validation_fold": validation_fold,
                    "accuracy_percent": accuracy * 100,
                    **param_dict
                }
                validation_df_list.append(validation_row)

        validation_df = pd.DataFrame(validation_df_list)
        best_param_dict = validation_df.groupby('n_neighbors')['accuracy_percent'].mean().idxmax()
        self.best_params_ = {"n_neighbors": best_param_dict}
        self.fit_one(self.best_params_, train_features, train_labels)

    def predict(self, test_features):
        return self.estimator.predict(test_features)



