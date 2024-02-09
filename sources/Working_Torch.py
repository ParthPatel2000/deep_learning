import numpy as np
from plotnine import *
import pandas as pd
import os
import urllib.request as download
import plotnine as p9
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


class TorchModel(nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()

        # Define the layers of the neural network
        layers = []
        for i in range(len(units_per_layer) - 1):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            layers.append(nn.ReLU())  # You can use other activation functions as well

        # Remove the last ReLU layer (if any) to have a continuous output
        if isinstance(layers[-1], nn.ReLU):
            layers.pop()

        # Create the neural network using the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Define the forward pass of the neural network
        # x = x.view(x.size(0), -1)  # Flatten the input if it's not already flat
        return self.model(x)


class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer

        # Instantiate TorchModel
        self.model = TorchModel(units_per_layer)

        # Instantiate optimizer and loss function
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.BCEWithLogitsLoss()

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
        if (type(X) == pd.DataFrame):
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            y_tensor = torch.tensor(y.values, dtype=torch.float32)
        else:
            X_tensor = X
            y_tensor = y

        if validation_data is not None:
            X_val, y_val = validation_data
            if (type(X_val) == pd.DataFrame):
                print(X_val.shape)
                X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
            else:
                X_val_tensor = X_val
                y_val_tensor = y_val

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
            if (type(X) == pd.DataFrame):
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
            else:
                X_tensor = X
            scores = self.model(X_tensor).numpy()
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        # Assuming binary classification with threshold at 0.5
        return (scores >= 0.5).astype(np.int32)


class TorchLearnerWithModelSelection:
    def __init__(self, max_epochs, batch_size, step_size, model_type, input_size):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.input_size = input_size
        self.model_type = model_type

        if self.model_type == "linear":
            self.model = LinearModel(input_size)
        elif self.model_type == "deep_neural_network":
            self.model = DeepNeuralNetworkModel(input_size)
        else:
            raise ValueError("Invalid model_type. Supported values: 'linear', 'deep_neural_network'")

        # Instantiate optimizer and loss function
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.BCEWithLogitsLoss()

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
        if type(X) == pd.DataFrame:
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            y_tensor = torch.tensor(y.values, dtype=torch.float32)
        else:
            X_tensor = X
            y_tensor = y

        if validation_data is not None:
            X_val, y_val = validation_data
            if (type(X_val) == pd.DataFrame):
                # print(X_val.shape)
                X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
            else:
                X_val_tensor = X_val
                y_val_tensor = y_val

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
            if (type(X) == pd.DataFrame):
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
            else:
                X_tensor = X
            scores = self.model(X_tensor).numpy()
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        # Assuming binary classification with threshold at 0.5
        return (scores >= 0.5).astype(np.int32)


class TorchLearnerCV:
    def __init__(self, units_per_layer, max_epochs=100, batch_size=32, step_size=0.01):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.model = TorchModel(units_per_layer)
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.BCEWithLogitsLoss()

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
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.max_epochs):
            train_losses = []
            for batch_features, batch_labels in train_loader:
                loss = self.take_step(batch_features, batch_labels)
                train_losses.append(loss)

            # Compute validation loss
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch_features, val_batch_labels in val_loader:
                    val_predictions = self.model(val_batch_features)
                    val_loss = self.loss_fun(val_predictions, val_batch_labels)
                    val_losses.append(val_loss.item())

            mean_val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch + 1}/{self.max_epochs} - Validation Loss: {mean_val_loss}")

            # Update best epoch if the current one has a lower validation loss
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_epoch = epoch

        print(f"Best epoch: {best_epoch + 1} with validation loss: {best_val_loss}")

        # Re-run gradient descent on the entire train set using the best number of epochs
        full_train_dataset = TensorDataset(torch.Tensor(X.values), torch.Tensor(y.values))
        full_train_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(best_epoch + 1):
            for full_train_batch_features, full_train_batch_labels in full_train_loader:
                self.take_step(full_train_batch_features, full_train_batch_labels)

    def decision_function(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()

    def predict(self, X, threshold=0.5):
        if not torch.is_tensor(X):
            X = torch.tensor(X.values, dtype=torch.float32)
        decision_function_output = self.decision_function(X)
        return (decision_function_output > threshold).astype(int)


class TorchLearnerCVWithModelSelection:
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
            self.model = DeepNeuralNetworkModel(units_per_layer[0])
        else:
            raise ValueError("Invalid model_type. Supported values: 'linear', 'deep_neural_network'")

        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.BCELoss()

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
            print(f"Epoch {epoch + 1}/{self.max_epochs} - Training Loss: {mean_train_loss}")
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
            print(f"Epoch {epoch + 1}/{self.max_epochs} - Validation Loss: {mean_val_loss}")

            # Update the best epoch if the current one has a lower validation loss
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_epoch = epoch

        print(f"Best epoch: {best_epoch + 1} with validation loss: {best_val_loss}")

        # Re-run gradient descent on the entire train set using the best number of epochs
        full_train_dataset = TensorDataset(torch.Tensor(X.values), torch.Tensor(y.values))
        full_train_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(best_epoch + 1):
            for full_train_batch_features, full_train_batch_labels in full_train_loader:
                self.take_step(full_train_batch_features, full_train_batch_labels)

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
        return (decision_function_output > threshold).astype(int)


class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.model = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_size):
        super(DeepNeuralNetworkModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)


def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = (X.sum(axis=1) + np.random.normal(0, noise, n_samples)) > 5
    return pd.DataFrame(X), pd.DataFrame(y, columns=['label'])


def calculate_accuracy(predictions, ground_truth):
    # Ensure predictions and ground_truth are NumPy arrays or lists
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Calculate the number of correct predictions
    correct_predictions = np.sum(predictions == ground_truth)

    # Calculate the total number of predictions
    total_predictions = len(predictions)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy


if __name__ == '__main__':

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

    accuracy_df = {}
    model_types = ["linear", "deep_neural_network"]
    best_epochs = {}

    ## splitting the data into train and test
    # np.random.seed(42)

    # Choose the percentage of data to be used for validation
    # validation_percentage = 0.3
    #
    # X_train, X_val, y_train, y_val = train_test_split(
    #     data_features, data_labels, test_size=validation_percentage, random_state=42
    # )
    ## Splitting ends

    X_synthetic, y_synthetic = generate_synthetic_data()

    # Splitting the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

    # Train the model
    torch_learner = TorchLearnerCVWithModelSelection(
        max_epochs=100, batch_size=10, step_size=0.001,
        units_per_layer=[X_train.shape[1], 1],
        model_type='linear'
    )

    # Train the model
    subtrain_losses, validation_losses = torch_learner.fit(X_train, y_train)

    # Print training losses during training
    print("Subtrain Losses During Training:", subtrain_losses)

    # Make predictions on the validation set
    predictions = torch_learner.predict(X_val)

    # Print predictions and ground truth for inspection
    # print("Predictions:", predictions)
    # print("Ground Truth:", y_val.values.flatten())
    print("Predictions:", predictions.flatten())
    ground_truth = y_val.values.flatten()
    print("Ground Truth (Binary):", y_val.astype(int).values.flatten())
    correct_predictions = (predictions == ground_truth).sum().item()
    total_samples = len(ground_truth)
    accuracy = correct_predictions / total_samples
    # accuracy = calculate_accuracy(predictions, ground_truth)
    print(f"Accuracy: {accuracy :.2f}%")
