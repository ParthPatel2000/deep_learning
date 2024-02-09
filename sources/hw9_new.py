from sklearn.metrics import mean_squared_error
import os
from plotnine import *
import torch
import requests
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import warnings
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoCV
from tabulate import tabulate
import matplotlib

warnings.filterwarnings('ignore')

matplotlib.use("agg")


class TorchModel(nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()

        # Define the layers of the neural network
        layers = []
        for i in range(len(units_per_layer) - 2):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            layers.append(nn.ReLU())  # Add activation function after each linear layer

        # Append the last linear layer without activation
        layers.append(nn.Linear(units_per_layer[-2], 1))

        # Create the neural network using the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Define the forward pass of the neural network
        return self.model(x)


class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, model_type, input_size):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.input_size = input_size
        self.model_type = model_type

        if self.model_type == "linear":
            self.model = LinearModel(input_size)
        elif self.model_type == "deep_neural_network":
            self.model = DeepNeuralNetworkModel(input_size, units_per_layer=[input_size, 100, 10, 1])
        else:
            raise ValueError("Invalid model_type. Supported values: 'linear', 'deep_neural_network'")

        # Instantiate optimizer and loss function
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.MSELoss()

    def take_step(self, X, y):
        # Forward pass
        predictions = self.model(X)

        # Compute the loss
        loss = self.loss_fun(predictions, y)

        # Zero the gradients, backward pass, and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, X, y, validation_data=None):
        # Convert input data to PyTorch tensors
        X_tensor = self.to_tensor(X)
        y_tensor = self.to_tensor(y)

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor = self.to_tensor(X_val)
            y_val_tensor = self.to_tensor(y_val)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(self.max_epochs):
            for i in range(0, len(X_tensor), self.batch_size):
                # Get the current batch
                X_batch = X_tensor[i:i + self.batch_size]
                y_batch = y_tensor[i:i + self.batch_size]

                # Take a step using the current batch
                self.take_step(X_batch, y_batch)

            # Compute and store training loss for this epoch
            train_predictions = self.model(X_tensor)
            train_loss = self.loss_fun(train_predictions, y_tensor)
            train_losses.append(train_loss.item())

            # Compute and store validation loss if validation data is provided
            if validation_data is not None:
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self.loss_fun(val_predictions, y_val_tensor)
                    val_losses.append(val_loss.item())

        return train_losses, val_losses

    def decision_function(self, X):
        with torch.no_grad():
            X = self.to_tensor(X)
            scores = self.model(X).numpy()
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return scores

    def to_tensor(self, X):
        if not torch.is_tensor(X):
            if type(X) == pd.DataFrame:
                X = torch.tensor(X.values, dtype=torch.float32)
            elif type(X) == np.ndarray:
                X = torch.tensor(X, dtype=torch.float32)
            else:
                raise ValueError("Invalid type for X. Supported types: pd.DataFrame, np.ndarray")
        return X


class TorchLearnerCV:
    def __init__(self, units_per_layer, max_epochs=100, batch_size=32, step_size=0.01,
                 model_type='deep_neural_network'):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.model_type = model_type

        if self.model_type == "linear":
            self.model = LinearModel(units_per_layer[0])
        elif self.model_type == "deep_neural_network":
            self.model = DeepNeuralNetworkModel(units_per_layer[0], units_per_layer)
        else:
            raise ValueError("Invalid model_type. Supported values: 'linear', 'deep_neural_network'")

        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.MSELoss()

    def to_tensor(self, X):
        if not torch.is_tensor(X):
            if type(X) == pd.DataFrame:
                X = torch.tensor(X.values, dtype=torch.float32)
            elif type(X) == np.ndarray:
                X = torch.tensor(X, dtype=torch.float32)
            else:
                raise ValueError("Invalid type for X. Supported types: pd.DataFrame, np.ndarray")
        return X

    def take_step(self, X, y):
        self.model.train()
        predictions = self.model(X)
        loss = self.loss_fun(predictions, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, X, y):
        # Split data into subtrain and validation sets
        subtrain_features, val_features, subtrain_labels, val_labels = train_test_split(X, y, test_size=0.2,
                                                                                        random_state=42)
        ## converting all the data to torch tensors
        subtrain_labels = self.to_tensor(subtrain_labels)
        val_labels = self.to_tensor(val_labels)
        subtrain_features = self.to_tensor(subtrain_features)
        val_features = self.to_tensor(val_features)

        train_dataset = TensorDataset(torch.Tensor(subtrain_features), torch.Tensor(subtrain_labels))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(val_features), torch.Tensor(val_labels))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_epoch = 0

        subtrain_losses = []
        validation_losses = []

        for epoch in range(self.max_epochs):
            train_losses = []
            for batch_features, batch_labels in train_loader:
                loss = self.take_step(batch_features, batch_labels)
                train_losses.append(loss)

            mean_train_loss = sum(train_losses) / len(train_losses)
            # print(f"Epoch {epoch + 1}/{self.max_epochs} - Training Loss: {mean_train_loss}")
            subtrain_losses.append(mean_train_loss)

            # Compute validation loss
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch_features, val_batch_labels in val_loader:
                    val_predictions = self.model(val_batch_features)
                    val_loss = self.loss_fun(val_predictions, val_batch_labels)
                    # print("Validation_loss: ",val_loss.item())
                    val_losses.append(val_loss.item())

            mean_val_loss = sum(val_losses) / len(val_losses)
            validation_losses.append(mean_val_loss)
            # print(f"Epoch {epoch + 1}/{self.max_epochs} - Validation Loss: {mean_val_loss}")

            # Update the best epoch if the current one has a lower validation loss
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_epoch = epoch

        print(f"Best epoch: {best_epoch + 1} with validation loss: {best_val_loss}")

        # # Re-run gradient descent on the entire train set using the best number of epochs
        # full_train_dataset = TensorDataset(self.to_tensor(X), self.to_tensor(y))
        # full_train_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, shuffle=True)
        #
        # for epoch in range(best_epoch + 1):
        #     for full_train_batch_features, full_train_batch_labels in full_train_loader:
        #         self.take_step(full_train_batch_features, full_train_batch_labels)

        return subtrain_losses, validation_losses

    def decision_function(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()

    def predict(self, X, threshold=0.5):
        if not torch.is_tensor(X):
            X = torch.tensor(X.values, dtype=torch.float32)
        decision_function_output = self.decision_function(X)
        return decision_function_output


class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.model = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.model(x)


class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_size, units_per_layer):
        super(DeepNeuralNetworkModel, self).__init__()

        # Ensure there are at least two layers (input and output)
        if len(units_per_layer) < 2:
            raise ValueError("units_per_layer must have at least two elements (input and output sizes)")

        # Create a list to store the layers
        layers = []

        # Add the input layer
        layers.append(nn.Linear(input_size, units_per_layer[1]))
        layers.append(nn.ReLU())

        # Add hidden layers
        for i in range(1, len(units_per_layer) - 2):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(nn.Linear(units_per_layer[-2], units_per_layer[-1]))

        # Create the model using Sequential
        # print("Layers: ", layers)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FeaturelessBaseline:
    def __init__(self):
        self.mean_label = None

    def fit(self, X_train, y_train):
        self.mean_label = np.mean(y_train)

    def predict(self, X_test):
        return np.repeat(self.mean_label, X_test.shape[0])


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
        return self.model.predict(X_test)


class LinearModelBaseline:
    def __init__(self):
        self.model = LassoCV()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


def experiment(X, y, model, cv):
    test_losses = []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_loss = mean_squared_error(y_test, predictions)
        test_losses.append(test_loss)

    return np.mean(test_losses)


# if __name__ == "__main__":

# Define data location info
data_location_info = {
    "forest_fire": {
        "url": 'https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/forestfires.csv',
        "local_path": 'D:/home/forest_fire.csv',
        "output_column": "area",
        "drop_columns": ["month", "day"]
    },
    "airfoil_self_noise": {
        "url": 'https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/airfoil_self_noise.tsv',
        "local_path": 'D:/home/airfoil_self_noise.tsv',
        "output_column": "Scaled_sound_pressure_level.decibels",
        "sep": '\t'
    }
}

data_dict = {}
loss_dict = {}

# Download and process the file

for dataset_name, dataset_location in data_location_info.items():
    local_path = dataset_location["local_path"]

    if not os.path.exists(local_path):
        url = dataset_location["url"]
        response = requests.get(url)

        if response.status_code == 200:
            with open(local_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {dataset_name} dataset from {url} to {local_path}")
        else:
            print(f"Failed to download {dataset_name} dataset from {url}")
    else:
        print(f"{dataset_name} dataset already exists at {local_path}")

    # Read the dataset
    sep = dataset_location.get("sep", ",")
    dataset_df = pd.read_csv(local_path, sep=sep)

    # Extract labels and features
    labels = dataset_df[dataset_location["output_column"]].values.reshape(-1, 1)

    # Drop non-numeric columns if specified
    if "drop_columns" in dataset_location:
        dataset_df = dataset_df.drop(columns=dataset_location["drop_columns"])

    features = dataset_df.drop(columns=[dataset_location["output_column"]])

    # scaling the data
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    features = X_scaler.fit_transform(features)
    labels = Y_scaler.fit_transform(labels)

    data_dict[dataset_name] = (pd.DataFrame(features), pd.DataFrame(labels))

# print(data_dict['forest_fire'][0])
for dataset_name, (X, y) in data_dict.items():
    print("Dataset: ", dataset_name, "X shape: ", X.shape)
    baseline_models = {
        "featureless": FeaturelessBaseline(),
        "nearest_neighbors": NearestNeighborsBaseline(),
        "linear_model": LinearModelBaseline()
    }

    torch_models = {
        "TorchLearnerCVLinear": TorchLearnerCV(units_per_layer=(X.shape[1], 1), max_epochs=100,
                                               batch_size=10, step_size=0.0001,
                                               model_type='linear'),
        "TorchLearnerCVDeep": TorchLearnerCV(units_per_layer=(X.shape[1], 100, 10, 1),
                                             max_epochs=100, batch_size=10, step_size=0.0001)
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    print("---------------------------------------------")
    print("Dataset: ", dataset_name)
    for model_name, model in baseline_models.items():
        loss = experiment(X, y, model, cv)
        print(f"{dataset_name} {model_name} Test Loss: {loss}")
        loss_dict[f"{dataset_name}_{model_name}"] = loss

    for model_name, model in torch_models.items():
        train_loss_dict = []
        test_losses = []
        for fold_id, (train_index, test_index) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            subtrain_losses, validation_losses = model.fit(X, y)
            train_loss_dict.append((subtrain_losses, validation_losses))

            predictions = model.predict(X_test)
            test_loss = mean_squared_error(y_test, predictions)
            test_losses.append(test_loss)
        loss = np.mean(test_losses)
        print(f"{dataset_name} {model_name} Test Loss: {loss}")
        loss_dict[f"{dataset_name}_{model_name}"] = loss

        df_list = []
        print(len(train_loss_dict[0][0]))
        for fold_id, (subtrain_losses, validation_losses) in enumerate(train_loss_dict):
            df_list.append(pd.DataFrame({
                "Fold": [fold_id + 1] * (len(subtrain_losses) + len(validation_losses)),
                "Epochs": list(range(1, len(subtrain_losses) + 1)) + list(range(1, len(validation_losses) + 1)),
                "Type": ["Subtrain Loss"] * len(subtrain_losses) + ["Validation Loss"] * len(validation_losses),
                "Loss": subtrain_losses + validation_losses
            }))

        df = pd.concat(df_list)
        # print(df)
        plot = (
                ggplot(df, aes(x='Epochs', y='Loss', color='Type')) +
                geom_line() +
                facet_wrap('~ Fold', scales='free') +
                labs(title="Subtrain and Validation Losses for Each Fold",
                     x="Epochs",
                     y="Loss")
        ).save(f"D:/home/plots/hw9/{dataset_name}_{model_name}.png")

    print("---------------------------------------------")
rows = list(loss_dict.items())

# Print the table
print(tabulate(rows, headers=["Model", "MSE"], tablefmt="grid"))
