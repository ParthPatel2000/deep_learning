import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from urllib import request as download
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib
import pandas as pd
import warnings
from plotnine import *
import torchsummary as summary

warnings.filterwarnings("ignore")
matplotlib.use('Agg')


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


class ConvolutionalMLP(nn.Module):
    def __init__(self, input_shape):
        super(ConvolutionalMLP, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[1], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(64 * input_shape[2] * input_shape[3], 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("x shape: ", x.shape)
        x = self.relu(self.conv1(x.unsqueeze(1).float()))
        # print("x conv1 output : ", x.shape)
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DenseMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DenseMLP, self).__init__()
        self.layers = nn.ModuleList()
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x.squeeze(1)))
        x = self.layers[-1](x)
        return x


class ConvolutionalMLPClassifier:
    def __init__(self, model, epochs=10, lr=0.001, batch_size=1, input_shape=(1, 16, 16)):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.input_shape = (batch_size, *input_shape)
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss_mean = []
        self.val_loss_mean = []

    def fit(self, X, y):
        # Convert to PyTorch tensors
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = to_tensor(X_train)
        X_test = to_tensor(X_test)
        y_train = to_tensor(y_train)
        y_test = to_tensor(y_test)

        # Create DataLoader
        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_test, y_test)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop
        mean_train_losses = []
        mean_val_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []
            for inputs, target in train_loader:
                # print("input shape: ", inputs.shape)
                # print("target shape: ", target.shape)
                self.optimizer.zero_grad()
                outputs = self.model.forward(inputs.unsqueeze(1).float())
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            mean_train_losses.append(np.mean(train_loss))

            self.model.eval()
            epoch_val_loss = []
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_outputs = self.model(val_inputs.unsqueeze(1).float())
                    val_loss = self.criterion(val_outputs, val_targets)
                    epoch_val_loss.append(val_loss.item())
            mean_val_losses.append(np.mean(epoch_val_loss))
            # print(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {np.mean(train_loss)} | Val Loss: {np.mean(epoch_val_loss)}")
        self.train_loss_mean.append(mean_train_losses)
        self.val_loss_mean.append(mean_val_losses)

    def predict(self, X):
        # Convert to PyTorch tensor
        # X_tensor = torch.tensor(X, dtype=torch.float32)
        X_tensor = to_tensor(X)

        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor.unsqueeze(1).float())

        # Convert outputs to probabilities using sigmoid activation
        predictions = torch.sigmoid(outputs).numpy()

        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)

        return binary_predictions


def experiment_loss(X, y, model, cv):
    subtrain_losses = []
    validation_losses = []
    accuracy = 0
    test_loss_fun = nn.BCEWithLogitsLoss()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for train_index, test_index in cv.split(X_train):
        subtrain_losses_per_fold = []
        validation_losses_per_fold = []
        X_subtrain, X_validation = X_train.iloc[train_index], X_train.iloc[test_index]
        y_subtrain, y_validation = y_train.iloc[train_index], y_train.iloc[test_index]

        X_subtrain = to_tensor(X_subtrain)
        X_validation = to_tensor(X_validation)
        y_subtrain = to_tensor(y_subtrain)
        y_validation = to_tensor(y_validation)

        model.fit(X_subtrain, y_subtrain)

        subtrain_predictions = to_tensor(model.predict(X_validation))
        subtrain_loss = test_loss_fun(subtrain_predictions, y_validation)
        subtrain_losses_per_fold.append(subtrain_loss)

        validation_predictions = to_tensor(model.predict(X_validation))
        validation_loss = test_loss_fun(validation_predictions, y_validation)
        validation_losses_per_fold.append(validation_loss)

    subtrain_losses.append(subtrain_losses_per_fold)
    validation_losses.append(validation_losses_per_fold)

    test_predictions = to_tensor(model.predict(to_tensor(X_test)))
    test_loss = test_loss_fun(test_predictions, to_tensor(y_test))

    subtrain_losses = np.mean(subtrain_losses, axis=0)
    validation_losses = np.mean(validation_losses, axis=0)

    accuracy = accuracy_score(y_test, test_predictions)

    return accuracy, test_loss, subtrain_losses, validation_losses


# data preprocessing
data_location_info = \
    {
        "zip_train":
            (
                "D:/home/zip.train.gz",
                "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/zip.train.gz"
            ),
        "zip_test":
            (
                "D:/home/zip.test.gz",
                "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/zip.test.gz"
            )
    }

data_info_dict = \
    {
        "zip_train": ("D:/home/zip.train.gz", 0),
        "zip_test": ("D:/home/zip.test.gz", 0),
    }

for file_name, (file_path, file_url) in data_location_info.items():
    if not os.path.isfile(file_path):
        download.urlretrieve(file_url, file_path)
        print(file_name, 'downloaded successfully')
    else:
        print(file_name, "Already exists")

data_dict = {}
for _ in range(2):
    for data_name, (file_name, label_col_num) in data_info_dict.items():
        data_df = pd.read_csv(file_name, sep=" ", header=None)
        label_col_num = int(label_col_num)  # Convert label_col_num to an integer
        data_label_vec = data_df.iloc[:, label_col_num]
        if _ == 0:
            is_01 = data_label_vec.isin([0, 1])
            data01_df = data_df[is_01]  # Use boolean indexing directly on data_df
        else:
            is_01 = data_label_vec.isin([7, 1])
            data01_df = data_df[is_01]  # Use boolean indexing directly on data_df
        is_label_col = data_df.columns == label_col_num
        data_features = data01_df.iloc[:, ~is_label_col]
        data_labels = data01_df.iloc[:, is_label_col]
        if data_name == "zip_train":
            data_features = data_features.iloc[:, :-1]
            data_df = data_df.iloc[:, 1:-1]
        if _ == 0:
            data_dict[f"{data_name}01"] = (data_features, data_labels)
        else:
            data_dict[f"{data_name}71"] = (data_features, data_labels)
print(data_dict.keys())
# data preprocessing done


baseline_models = {
    "featureless": FeaturelessBaseline(),
    "nearest_neighbors": NearestNeighborsBaseline(),
    "linear_model": LinearModelBaseline()
}

batch_size = 64
epochs = 20
dense_hidden_sizes = [256, 512, 256, 128]
flattened_size = 16 * 16
dense_mlp = DenseMLP(input_size=flattened_size,
                     hidden_sizes=dense_hidden_sizes,
                     output_size=1)
convolutional_mlp = ConvolutionalMLP(input_shape=(batch_size, 1, 16, 16))
convolutional_models = {

    'Convolutional': ConvolutionalMLPClassifier(model=convolutional_mlp,
                                                lr=0.0001,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                input_shape=(1, 16, 16)),
    'Dense': ConvolutionalMLPClassifier(model=dense_mlp,
                                        lr=0.0001,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        input_shape=(1, 16, 16))

}
cv = KFold(n_splits=3, shuffle=True, random_state=42)

accuracy_dict = {}
param_grid = []
loss_df = pd.DataFrame()
loss_dict = {}

for dataset_name, (X, y) in data_dict.items():
    print(f"Dataset: {dataset_name}")
    loss_mean_df = pd.DataFrame()
    for model_name, model in baseline_models.items():
        accuracy, loss, subtrain_loss, validation_loss = experiment_loss(X, y, model, cv)

        print("Model: ", model_name)
        print(f"{dataset_name}_{model_name} Test Loss: {loss}")
        # print(f"{dataset_name}_{model_name} Test Accuracy: {accuracy}")
        print(f"{dataset_name}_{model_name} Subtrain Loss: {subtrain_loss}")
        print(f"{dataset_name}_{model_name} Validation Loss: {validation_loss}")

        loss_dict[f"{dataset_name}_{model_name}"] = loss
        accuracy_dict[f"{dataset_name}_{model_name}"] = accuracy

for model_name, model_instance in convolutional_models.items():
    print(f"Model: {model_name}")
    loss_mean_df = pd.DataFrame()
    for dataset_name, (X, y) in data_dict.items():
        print(f"Dataset: {dataset_name}")
        for fold_id, (train_index, test_index) in enumerate(cv.split(X)):
            subtrain_losses_per_fold = []
            validation_losses_per_fold = []
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # creating tensors
            X_train = to_tensor(X_train)
            X_test = to_tensor(X_test)
            y_train = to_tensor(y_train)
            y_test = to_tensor(y_test)

            # fitting the model
            model_instance.fit(X_train, y_train)

        # predicting
        predictions = to_tensor(model_instance.predict(X))

        # calculating accuracy
        accuracy_dict[f"{dataset_name}_{model_name}"] = accuracy_score(y, predictions)

        # retrieving loss values
        mean_train_losses = np.mean(model_instance.train_loss_mean, axis=0)
        std_train_losses = np.std(model_instance.train_loss_mean, axis=0)
        mean_val_losses = np.mean(model_instance.val_loss_mean, axis=0)
        std_val_losses = np.std(model_instance.val_loss_mean, axis=0)
        loss_dict[f"{dataset_name}_{model_name}"] = np.mean(mean_val_losses)
        new_df = pd.DataFrame({
            'epoch': np.arange(1, len(mean_train_losses) + 1),
            'model': model_name,
            'dataset': dataset_name,
            'train_loss': mean_train_losses,
            'train_loss_std': std_train_losses,
            'validation_loss': mean_val_losses,
            'validation_loss_std': std_val_losses
        })
        loss_mean_df = pd.concat([loss_mean_df, new_df], axis=0, ignore_index=True)
        loss_df = pd.concat([loss_df, loss_mean_df], axis=0, ignore_index=True)
    ## plotting
    plot = (ggplot(loss_mean_df, aes(x='epoch')) +
            geom_line(aes(x='epoch', y='validation_loss'), color='red', size=1) +
            geom_line(aes(x='epoch', y='train_loss'), color='blue', size=1) +
            geom_line(aes(x='epoch', y='validation_loss'), color='red', size=1) +
            geom_line(aes(x='epoch', y='train_loss'), color='blue', size=1) +
            scale_x_continuous(breaks=range(1, len(loss_mean_df['epoch']) + 1)) +
            labs(title=f'Training and Validation Loss - {model_name}',
                 x='Epoch',
                 y='Mean Loss') +
            facet_wrap('~dataset')).save(f"D:/home/plots/hw13_{model_name}_loss_plot.png")
    # print(loss_mean_df)

accuracy_table = pd.DataFrame(list(accuracy_dict.items()), columns=['Model', 'Test Accuracy'])
print(loss_df)
print()
print(accuracy_table)
