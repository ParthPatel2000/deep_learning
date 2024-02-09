import os
import urllib.request as download
import warnings
import time
from plotnine import *
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from plotnine import geom_line, geom_point, scale_color_manual
from plotnine import ggplot, aes, labs
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


class FeaturelessBaseline:
    def __init__(self):
        self.most_common_label = None

    def fit(self, X_train, y_train):
        y_train_int = y_train.numpy().flatten().astype(np.int64)
        self.most_common_label = np.argmax(np.bincount(y_train_int))

    def predict(self, X_test):
        return to_tensor(np.full(X_test.shape[0], self.most_common_label)).reshape(X_test.shape[0], 1)


class NearestNeighborsBaseline:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train):
        # You can use GridSearchCV to find the best hyperparameters
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        grid_search = GridSearchCV(self.model, param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        # Set the model with the best hyperparameters
        self.model = grid_search.best_estimator_

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        predictions = np.round(predictions).astype(np.float32)
        return predictions


class LinearModelBaseline:
    def __init__(self):
        self.model = LogisticRegressionCV(cv=3)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test).reshape(X_test.shape[0], 1)


# The Model
class MLP(nn.Module):
    def __init__(self, units_per_layer):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(units_per_layer) - 1):
            self.layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            if i < len(units_per_layer) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#
class OptimizerMLP:
    def __init__(self, max_epochs, units_per_layer):
        self.max_epochs = max_epochs
        self.units_per_layer = units_per_layer
        self.opt_name = None
        self.opt_params = None
        self.model = MLP(units_per_layer)
        # self.epoch_losses = pd.DataFrame(columns=['epoch', 'loss', 'opt_name', 'opt_params'])

    def fit(self, subtrain_features, subtrain_labels, learning_rate=0.001):
        self.opt_params["lr"] = learning_rate
        criterion = nn.BCEWithLogitsLoss()
        optimizer = self._get_optimizer()

        # Training loop
        for epoch in range(self.max_epochs):
            # Forward pass
            outputs = self.model(subtrain_features)
            loss = criterion(outputs, subtrain_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculate and print loss (you can save it for plotting)
            # print(f'Epoch [{epoch + 1}/{self.max_epochs}], Loss: {loss.item()}')

    def predict(self, test_features):
        # Implement prediction logic
        test_features = to_tensor(test_features)

        with torch.no_grad():
            scores = self.model(test_features)
            predictions = torch.sigmoid(scores).numpy()
        # print(predictions.shape)
        return predictions

    def _get_optimizer(self):
        if self.opt_name == "SGD":
            return optim.SGD(self.model.parameters(), **self.opt_params)
        elif self.opt_name == "Adam":
            return optim.Adam(self.model.parameters(), **self.opt_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt_name}")


class MyCV:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.loss_each_fold = pd.DataFrame(columns=['fold', 'set', 'epoch', 'loss', 'opt_name', 'opt_params'])
        self.loss_mean = pd.DataFrame(columns=['set', 'epoch', 'loss', 'opt_name', 'opt_params'])
        self.best_param = None
        self.learning_rate = 0.001

    def fit(self, train_features, train_labels):

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        epoch_losses = []
        for fold_id, (train_index, val_index) in enumerate(kf.split(train_features)):
            subtrain_features, subval_features = train_features.iloc[train_index], train_features.iloc[val_index]
            subtrain_labels, subval_labels = train_labels.iloc[train_index], train_labels.iloc[val_index]

            subtrain_features = to_tensor(subtrain_features)
            subtrain_labels = to_tensor(subtrain_labels)
            subval_features = to_tensor(subval_features)
            subval_labels = to_tensor(subval_labels)

            for param_set in self.param_grid:
                self.estimator.opt_name = param_set["opt_name"]
                self.estimator.opt_params = param_set["opt_params"]

                # Call fit on subtrain data
                self.estimator.fit(subtrain_features, subtrain_labels, learning_rate=self.learning_rate)

                # Calculate zero-one loss on subtrain/validation data
                loss_func = nn.BCEWithLogitsLoss()
                predictions = self.estimator.predict(subtrain_features)
                validation_predictions = self.estimator.predict(subval_features)

                subtrain_loss = loss_func(to_tensor(predictions), subtrain_labels)
                subval_loss = loss_func(to_tensor(validation_predictions), subval_labels)

                new_row = {
                    'fold': len(self.loss_each_fold) + 1,
                    'set': 'subtrain',
                    'epoch': self.estimator.max_epochs,
                    'loss': subtrain_loss,
                    'opt_name': self.estimator.opt_name,
                    'opt_params': self.estimator.opt_params
                }

                self.loss_each_fold = pd.concat([self.loss_each_fold, pd.DataFrame([new_row])], ignore_index=True)

                new_row = {
                    'fold': len(self.loss_each_fold) + 1,
                    'set': 'subval',
                    'epoch': self.estimator.max_epochs,
                    'loss': subval_loss,
                    'opt_name': self.estimator.opt_name,
                    'opt_params': self.estimator.opt_params
                }

                self.loss_each_fold = pd.concat([self.loss_each_fold, pd.DataFrame([new_row])], ignore_index=True)
            # print("length of epoch losses:", len(epoch_losses))
            # Calculate mean loss over folds
            mean_subtrain_loss = self.loss_each_fold[self.loss_each_fold['set'] == 'subtrain']['loss'].mean()
            mean_subval_loss = self.loss_each_fold[self.loss_each_fold['set'] == 'subval']['loss'].mean()

        new_row_subtrain = {
            'set': 'subtrain',
            'epoch': self.estimator.max_epochs,
            'loss': mean_subtrain_loss.item(),
            'opt_name': self.estimator.opt_name,
            'opt_params': self.estimator.opt_params
        }

        new_row_subval = {
            'set': 'subval',
            'epoch': self.estimator.max_epochs,
            'loss': mean_subval_loss.item(),
            'opt_name': self.estimator.opt_name,
            'opt_params': self.estimator.opt_params
        }

        self.loss_mean = pd.concat([self.loss_mean, pd.DataFrame([new_row_subtrain, new_row_subval])],
                                   ignore_index=True)

        # Find the best hyper-parameters
        best_param_index = self.loss_mean['loss'].idxmin()
        self.best_param = {
            'opt_name': self.loss_mean.loc[best_param_index, 'opt_name'],
            'opt_params': self.loss_mean.loc[best_param_index, 'opt_params']
        }

        # Set attributes of estimator with best_param
        self.estimator.opt_name = self.best_param['opt_name']
        self.estimator.opt_params = self.best_param['opt_params']

        # Call fit on the entire training set
        train_features = to_tensor(train_features)
        train_labels = to_tensor(train_labels)
        self.estimator.fit(train_features, train_labels)

    def predict(self, test_features):
        test_features = to_tensor(test_features)

        # Call the predict method of the estimator
        predictions = self.estimator.predict(test_features)

        return predictions


def to_tensor(X):
    if not torch.is_tensor(X):
        if type(X) == pd.DataFrame:
            X = torch.tensor(X.values, dtype=torch.float32)
        elif type(X) == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError("Invalid type for X. Supported types: pd.DataFrame, np.ndarray")
    return X


def experiment_loss(X, y, model, cv):
    test_losses = []
    accuracy_values = []
    test_loss_fun = nn.BCEWithLogitsLoss()

    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = to_tensor(X_train)
        X_test = to_tensor(X_test)
        y_train = to_tensor(y_train)
        y_test = to_tensor(y_test)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = to_tensor(predictions)
        # print(predictions.numpy().flatten())
        # print(y_test.numpy().flatten())

        test_loss = test_loss_fun(predictions, y_test)
        test_losses.append(test_loss)

        accuracy = accuracy_score(y_test, predictions)
        accuracy_values.append(accuracy)

    return np.mean(test_losses), np.mean(accuracy_values)


##Data Preprocessing
data_location_info = \
    {
        "zip":
            (
                "D:/home/zip.train.gz",
                "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/zip.train.gz"
            ),
        "spam":
            (
                "D:/home/spam.train.gz",
                "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/spam.data"
            )
    }

data_info_dict = \
    {
        "zip": ("D:/home/zip.train.gz", 0),
        "spam": ("D:/home/spam.data", 0),
    }

for file_name, (file_path, file_url) in data_location_info.items():
    if not os.path.isfile(file_path):
        download.urlretrieve(file_url, file_path)
        print(file_name, 'downloaded successfully')
    else:
        print(file_name, "Already exists")

data_dict = {}

for data_name, (file_name, label_col_num) in data_info_dict.items():
    data_df = pd.read_csv(file_name, sep=" ", header=None)
    # print(data_name," : ",data_df.shape)
    label_col_num = int(label_col_num)  # Convert label_col_num to an integer
    data_label_vec = data_df.iloc[:, label_col_num]
    is_01 = data_label_vec.isin([0, 1])
    data01_df = data_df[is_01]  # Use boolean indexing directly on data_df
    is_label_col = data_df.columns == label_col_num
    data_features = data01_df.iloc[:, ~is_label_col]
    data_labels = data01_df.iloc[:, is_label_col]
    if (data_name == "zip"):
        data_features = data_features.iloc[:, :-1]
        data_df = data_df.iloc[:, 1:-1]
        # print(data_df.shape)
    data_dict[data_name] = (data_features, data_labels)
    # print(type(data_features), type(data_labels))

spam_features, spam_labels = data_dict.pop("spam")
n_spam_rows, n_spam_features = spam_features.shape
spam_mean = spam_features.mean().to_numpy().reshape(1, n_spam_features)
spam_std = spam_features.std().to_numpy().reshape(1, n_spam_features)
spam_scaled = (spam_features - spam_mean) / spam_std
data_dict["spam_scaled"] = (spam_scaled, spam_labels)

## Data Preprocessing ends

loss_dict = {}
accuracy_dict = {}
param_grid = []
loss_mean_df = None
loss_each_fold_df = None

for momentum in 0.1, 0.5:
    for learning_rate in 0.0001, 0.001, 0.01:
        param_grid.append({
            "opt_name": "SGD",
            "opt_params": {"momentum": momentum, "lr": learning_rate}
        })

for beta1 in 0.85, 0.9, 0.95:
    for beta2 in 0.99, 0.999, 0.9999:
        param_grid.append({
            "opt_name": "Adam",
            "opt_params": {"betas": (beta1, beta2)}
        })

for dataset_name, (X, y) in data_dict.items():
    print("Dataset: ", dataset_name, "X shape: ", X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    baseline_models = {
        "featureless": FeaturelessBaseline(),
        "nearest_neighbors": NearestNeighborsBaseline(),
        "linear_model": LinearModelBaseline()
    }
    units_per_layer = (X.shape[1], 100, 100, 1)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    omlp = OptimizerMLP(max_epochs=100, units_per_layer=units_per_layer)
    learner_instance = MyCV(
        estimator=omlp,
        param_grid=param_grid,
        cv=3)

    print("---------------------------------------------")
    print("Dataset: ", dataset_name)
    loss_mean_dict = {}
    # accuracy_dict = {}

    for model_name, model in baseline_models.items():
        print("Model: ", model_name)
        loss, accuracy = experiment_loss(X_scaled_df, y, model, cv)
        print(f"{dataset_name}_{model_name} Test Loss: {loss}")
        print(f"{dataset_name}_{model_name} Test Accuracy: {accuracy}")
        loss_dict[f"{dataset_name}_{model_name}"] = loss
        accuracy_dict[f"{dataset_name}_{model_name}"] = accuracy

    for fold_id, (train_index, test_index) in enumerate(cv.split(X_scaled_df)):
        test_losses = []
        X_train, X_test = X_scaled_df.iloc[train_index], X_scaled_df.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        learner_instance.fit(X_train, y_train)

        predictions = learner_instance.predict(X_test)

        loss_func = nn.BCEWithLogitsLoss()
        test_loss = loss_func(to_tensor(predictions), to_tensor(predictions))
        test_losses.append(test_loss)
        predictions = np.round(predictions).astype(np.float32)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_dict[f"{dataset_name}_omlp"] = accuracy

        loss = np.mean(test_losses)
        print(f"{dataset_name} omlp Test Loss: {loss}")
        print(f"{dataset_name} omlp Test Accuracy: {accuracy}")
        loss_dict[f"{dataset_name}_omlp"] = loss

    # print('loss_each_fold\n', learner_instance.loss_each_fold.head())
    # print("loss mean\n", learner_instance.loss_mean.head())
    # print("Loss mean shape: ", learner_instance.loss_mean.shape)
    #
    # print("columns of loss mean\n", learner_instance.loss_mean.columns)
    # print("columns of loss each fold\n", learner_instance.loss_each_fold.columns)
    #

    colors = {'subtrain': 'red', 'validation': 'blue'}

    # Plot for spam dataset
    loss_mean = learner_instance.loss_mean
    spam_plot = (
            ggplot(loss_mean, aes(x='epoch', y='loss', color='set')) +
            geom_line() +
            geom_point(
                data=loss_mean.loc[
                    loss_mean.groupby(['set'])['loss'].idxmin()],
                size=3
            ) +
            facet_wrap('opt_name') +
            labs(title=f'{dataset_name} Loss', x='Epochs', y='Loss') +
            scale_color_manual(values=colors)
    ).save(f"D:/home/plots/hw12_{dataset_name}_plot.png")

loss_table = pd.DataFrame(list(loss_dict.items()), columns=['Model', 'Test Loss'])
accuracy_table = pd.DataFrame(list(accuracy_dict.items()), columns=['Model', 'Test Accuracy'])
print(loss_table)
print()
print(accuracy_table)
