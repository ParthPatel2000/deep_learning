from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from plotnine import *
import pandas as pd
import os
import urllib.request as download
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def to_tensor(X):
    if not torch.is_tensor(X):
        if type(X) == pd.DataFrame:
            X = torch.tensor(X.values, dtype=torch.float32)
        elif type(X) == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError("Invalid type for X. Supported types: pd.DataFrame, np.ndarray")
    return X


class FeaturelessBaseline:
    def __init__(self):
        self.most_common_label = None

    def fit(self, X_train, y_train):
        y_train_int = y_train.to_numpy().flatten().astype(np.int64)
        self.most_common_label = np.argmax(np.bincount(y_train_int))

    def predict(self, X_test):
        return to_tensor(np.full(X_test.shape[0], self.most_common_label)).reshape(X_test.shape[0], 1)


class RegularizedMLP(nn.Module):
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer, hidden_layers=5):
        super(RegularizedMLP, self).__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.hidden_layers = hidden_layers
        self.hidden_units = 10

        # Dynamic creation of hidden layers based on max_hidden_layers
        layers = []
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.ReLU())

        layers.insert(0, nn.Linear(units_per_layer[0], self.hidden_units))
        layers.append(nn.Linear(self.hidden_units, units_per_layer[1]))

        # layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, input_matrix):
        return self.network(input_matrix)


class RegularizedMLPLearner:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer, hidden_layers=1):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.hidden_layers = hidden_layers
        self.model = None
        self.optimizer = None
        self.loss_fun = nn.BCEWithLogitsLoss()
        self.subtrain_losses = []
        self.val_losses = []

    def set_params(self, **params):
        self.hidden_layers = params['hidden_layers']
        self.model = RegularizedMLP(max_epochs=self.max_epochs, batch_size=self.batch_size,
                                    step_size=self.step_size, units_per_layer=self.units_per_layer,
                                    hidden_layers=self.hidden_layers)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.step_size)

    def take_step(self, X, y):
        predictions = self.model(X)
        loss = self.loss_fun(predictions, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, X, y, validation_data=None):
        X = to_tensor(X)
        y = to_tensor(y)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        validation_dataset = torch.utils.data.TensorDataset(validation_data[0], validation_data[1])
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size,
                                                            shuffle=True)

        epoch_subtrain_loss = []
        epoch_val_loss = []
        for epoch in range(self.max_epochs):
            for batch_features, batch_labels in dataloader:
                self.take_step(batch_features, batch_labels)
                epoch_subtrain_loss.append(self.loss_fun(self.model(batch_features), batch_labels).item())

            if validation_data is not None:
                with torch.no_grad():
                    for batch_features, batch_labels in validation_dataloader:
                        epoch_val_loss.append(self.loss_fun(self.model(batch_features), batch_labels).item())

        self.subtrain_losses.append(np.mean(epoch_subtrain_loss))
        self.val_losses.append(np.mean(epoch_val_loss))

    def predict(self, X):
        X = to_tensor(X)
        with torch.no_grad():
            predicted_classes = self.model(X)
            return predicted_classes


class MyCV:
    def __init__(self, estimator, param_grid, cv=5):
        self.loss_each_fold = pd.DataFrame()
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.loss_mean = None
        self.subtrain_loss = []
        self.val_loss = []
        self.best_param = None

    def fit(self, X, y):
        fold_losses = pd.DataFrame()
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        for param_set in self.param_grid:
            self.estimator.set_params(**param_set)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                subtrain_features, subtrain_labels = X.iloc[train_idx], y.iloc[train_idx]
                val_features, val_labels = X.iloc[val_idx], y.iloc[val_idx]

                subtrain_features = to_tensor(subtrain_features)
                subtrain_labels = to_tensor(subtrain_labels)
                val_features = to_tensor(val_features)
                val_labels = to_tensor(val_labels)

                self.estimator.fit(subtrain_features, subtrain_labels, validation_data=(val_features, val_labels))
                self.subtrain_loss.append(self.estimator.subtrain_losses)

                # Compute zero-one loss on validation set
                self.val_loss.append(self.estimator.val_losses)

                fold_losses = pd.concat([fold_losses,
                                         pd.DataFrame({'set': 'subtrain',
                                                       'hidden_layers': self.param_grid,
                                                       'loss': np.mean(self.estimator.subtrain_losses)})],
                                        ignore_index=True)

            self.loss_each_fold = fold_losses
            print("subtrain loss: ", np.mean(np.mean(self.subtrain_loss, axis=1)))
            print("loss each fold: ", self.loss_each_fold)
            self.loss_mean = self.loss_each_fold.groupby(['set', 'hidden_layers']).mean().reset_index()
            best_params_index = self.loss_mean['loss'].idxmin()
            keys_to_use = list(self.param_grid[0].keys())
            self.best_param = self.loss_mean.loc[best_params_index, keys_to_use]

            # Set the best hyperparameters to the estimator
            self.estimator.set_params(**self.best_param)
            self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def decision_function(self, X):
        return self.estimator.decision_function(X)


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


def evaluate_model(model, train_features, train_labels, test_features, test_labels):
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


# Define models and parameters
models = {
    'Featureless': FeaturelessBaseline(),
    'GridSearchCV+KNeighborsClassifier': GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': [1, 3, 5]}),
    'LogisticRegressionCV': LogisticRegressionCV(),
    'MyCV+RegularizedMLP': None
}

# Run experiments

print("Running experiments...")

accuracies = {'Algorithm': [], 'Test Accuracy': []}

input_dict = data_dict
kf = KFold(n_splits=2, shuffle=True, random_state=42)
for data_set_name, (features, labels) in input_dict.items():
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    print(f"Processing dataset: {data_set_name}")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features)):
        # Split the data into train and test sets
        train_features, train_labels = features.iloc[train_idx], labels.iloc[train_idx]
        test_features, test_labels = features.iloc[test_idx], labels.iloc[test_idx]

        for model_name, model_instance in models.items():
            print(f"Training model: {model_name}")
            if model_name != 'MyCV+RegularizedMLP':
                model_instance.fit(train_features, train_labels)
                # test_features, test_labels = input_dict[data_set_name]
                test_accuracy = accuracy_score(test_labels, model_instance.predict(test_features))


            else:
                model_instance = MyCV(estimator=RegularizedMLPLearner(max_epochs=20,
                                                                      batch_size=64,
                                                                      step_size=0.01,
                                                                      units_per_layer=[features.shape[1], 1]),
                                      param_grid=[{'hidden_layers': L} for L in range(1, 11)], cv=2)

                print(f"Processing fold: {fold_idx}")
                model_instance.fit(train_features, train_labels)

                # Evaluate the model on the test set
                # test_features, test_labels = input_dict[data_set_name]
                # test_labels = torch.relu(to_tensor(test_labels))

                predictions = model_instance.predict(test_features)
                predictions = torch.relu(predictions)

                predictions = (predictions >= 0.5).float()

                test_accuracy = accuracy_score(test_labels, predictions)

            accuracies['Algorithm'].append(f'{model_name}_{data_set_name}_fold_{fold_idx}')
            accuracies['Test Accuracy'].append(test_accuracy)

# Create a DataFrame for accuracies
accuracies = pd.DataFrame(accuracies)
average_accuracies = accuracies.groupby(['Algorithm'])['Test Accuracy'].mean().reset_index()
accuracy_df = pd.DataFrame(average_accuracies)

print(accuracy_df)

output_file_path = "D:/home/plots/hw11.png"

# Create a ggplot
plot = (
        ggplot(accuracy_df, aes(x='Algorithm', y='Test Accuracy'))
        + geom_line(color='blue')  # Adjust color if needed
        + labs(title='Test Accuracy of Different Algorithms', x='Algorithm', y='Test Accuracy')
)
plot.save(output_file_path)
