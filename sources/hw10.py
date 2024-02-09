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
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoCV
from tabulate import tabulate
import matplotlib
from torchvision import datasets, transforms

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
        layers.append(nn.Linear(units_per_layer[-2], units_per_layer[-1]))

        # Create the neural network using the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Define the forward pass of the neural network
        return self.model(x)


class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.input_size = self.units_per_layer[0]
        self.output_size = self.units_per_layer[-1]  # Number of classes for multi-class classification
        self.model = TorchModel(units_per_layer=self.units_per_layer)

        # Instantiate optimizer and loss function
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.CrossEntropyLoss()
        self.best_epochs = None
        self.loss_df = None

    def take_step(self, X, y):
        # Forward pass
        predictions = self.model(X)

        # Compute the loss
        min_label = y.min().item()
        y = y - min_label
        y = y.long()

        loss = self.loss_fun(predictions, y)

        # Zero the gradients, backward pass, and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError("Invalid type for X. Supported types: pd.DataFrame, np.ndarray")
        return X

    def fit(self, X, y, validation_data=None):
        """Gradient descent learning of weights"""
        train_dataset = torch.utils.data.TensorDataset(self.to_tensor(X), self.to_tensor(y))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = torch.utils.data.TensorDataset(self.to_tensor(X_val), self.to_tensor(y_val))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        loss_df_list = []
        best_epoch = 0
        best_val_loss = float('inf')

        for epoch in range(self.max_epochs):
            for batch_features, batch_labels in train_loader:
                self.take_step(batch_features, batch_labels)

            # Compute and store training loss for this epoch
            with torch.no_grad():
                train_predictions = self.model(self.to_tensor(X))
                train_loss = self.loss_fun(train_predictions, self.to_tensor(y).long())
                loss_df_list.append(
                    pd.DataFrame({"Epoch": epoch + 1, "Training Loss": train_loss.item()}, index=[epoch + 1]))

            # Compute and store validation loss if validation data is provided
            if validation_data is not None:
                with torch.no_grad():
                    val_predictions = self.model(self.to_tensor(X_val))
                    val_loss = self.loss_fun(val_predictions, self.to_tensor(y_val).long())
                    loss_df_list[-1]["Validation Loss"] = val_loss.item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1

        self.loss_df = pd.concat(loss_df_list)
        self.best_epochs = best_epoch

    def decision_function(self, X):
        with torch.no_grad():
            X = self.to_tensor(X)
            scores = self.model(X).numpy()
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return scores


class TorchLearnerCV:
    # def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
    #     self.subtrain_learner = TorchLearner(TODO)
    #
    # def fit(self, X, y):
    #     """cross-validation for selecting the best number of epochs"""
    #     self.subtrain_learner.validation_data = TODO
    #     self.subtrain_learner.fit(TODO_SUBTRAIN_DATA)
    #     self.train_learner = TorchLearner(max_epochs=best_epochs)
    #     self.train_learner.fit(TODO_TRAIN_DATA)
    #
    # def predict(self, X):
    #     self.train_learner.predict(X)

    class TorchLearnerCV:
        def __init__(self, max_epochs, batch_size, step_size, units_per_layer, num_folds=3):
            # Initialize the subtrain_learner with max_epochs
            self.units_per_layer = units_per_layer
            self.batch_size = batch_size
            self.step_size = step_size
            self.max_epochs = max_epochs
            self.num_folds = num_folds
            self.best_epochs = None
            self.cv_results = []

            self.subtrain_learner = TorchLearner(max_epochs, batch_size, step_size, units_per_layer)

            # Initialize the train_learner
            self.final_model = None

        def fit(self, X, y):
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

            best_epoch_sum = 0
            for fold, (train_index, val_index) in enumerate(kf.split(X)):
                print(f"Training Fold {fold + 1}/{self.num_folds}")

                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                subtrain_learner = TorchLearner(self.max_epochs, self.batch_size, self.step_size,
                                                self.units_per_layer)
                subtrain_learner.fit(X_train, y_train, validation_data=(X_val, y_val))
                best_epoch_sum += subtrain_learner.best_epochs

                self.final_model = TorchLearner(subtrain_learner.best_epochs, self.batch_size, self.step_size,
                                                self.units_per_layer)
                self.final_model.fit(X, y)

                # Evaluate on validation set
                val_predictions = self.final_model.predict(X_val)
                val_accuracy = accuracy_score(y_val, np.argmax(val_predictions, axis=1))
                self.cv_results.append({"Fold": fold + 1, "Validation Accuracy": val_accuracy})

                # Retrain on the entire training set
            # Calculate the average best number of epochs across folds
            self.best_epochs = int(np.round(best_epoch_sum / self.num_folds))
            self.final_model = TorchLearner(np.mean(self.best_epochs), self.batch_size, self.step_size,
                                            self.units_per_layer)
            self.final_model.fit(X, y)

        def predict(self, X):
            # Predict using the train_learner
            return self.train_learner.predict(X)


def test_model():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of samples
    num_samples = 1000

    # Number of features
    num_features = 20

    # Number of classes
    num_classes = 5

    # Generate random features
    X = np.random.randn(num_samples, num_features)

    # Generate random labels (integers from 0 to num_classes - 1)
    y = np.random.randint(0, num_classes, size=num_samples)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to Pandas DataFrames
    train_df = pd.DataFrame(data=X_train, columns=[f"feature_{i}" for i in range(num_features)])
    train_df["label"] = y_train

    val_df = pd.DataFrame(data=X_val, columns=[f"feature_{i}" for i in range(num_features)])
    val_df["label"] = y_val

    output_size = len(set(y_train))

    # Create the learner
    learner = TorchLearner(max_epochs=1000, batch_size=10, step_size=0.001,
                           units_per_layer=[num_features, 100, 100, 10, output_size])
    loss_df = learner.fit(X_train, y_train, validation_data=(X_val, y_val))
    # print(learner.loss_df.head())
    predictions = learner.predict(val_df.drop(columns=["label"]))
    plot = (
            ggplot(learner.loss_df, aes(x='Epoch'))
            + geom_line(aes(y='Training Loss'), color='blue', size=1, linetype='solid')
            + geom_line(aes(y='Validation Loss'), color='red', size=1, linetype='solid')
            + geom_vline(xintercept=learner.best_epochs, color='black', size=1, linetype='dashed')
            + labs(title='Training and Validation Loss Over Epochs', x='Epoch', y='Loss')
    ).save("D:/home/plots/loss.png")
    # accuracy
    # accuracy = accuracy_score(y_val, predictions.argmax(axis=1))
    accuracy = np.mean(y_val == predictions.argmax(axis=1))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Best epoch: {learner.best_epochs}")


def main():
    data_dict = {
        "zip": (FEATURES, LABELS),
        "MNIST": (FEATURES, LABELS)}
    test_error_df_list = []
    for data_name, TODO in data_dict.items():
        model_units = {
            "linear": (ncol, n_classes),
            "deep": (ncol, 100, 10, n_classes)
        }
        for test_fold, indices in enumerate(kf.split(TODO)):
            for model_name, units_per_layer in model_units.items():
                "fit(train data), then predict(test data), then store test error"
                test_error_df_list.append(test_row)
    test_error_df = pd.concat(test_error_df_list)
    p9.ggplot() + TODO


if __name__ == '__main__':
    test_model()
