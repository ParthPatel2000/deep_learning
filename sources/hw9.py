import matplotlib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
import os
import requests
from plotnine import *
import warnings

warnings.filterwarnings('ignore')
matplotlib.use("agg")


# Define the TorchModel class
class TorchModel(nn.Module):
    def __init__(self, *layer_units):
        super(TorchModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_units) - 1):
            self.layers.append(nn.Linear(layer_units[i], layer_units[i + 1]))

    def forward(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor


# Define the TorchLearner class
class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, layer_units):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.layers = []
        if layer_units:
            for i in range(len(layer_units) - 1):
                self.layers.append(nn.Linear(layer_units[i], layer_units[i + 1]))
            self.model = nn.Sequential(*self.layers)
            self.loss_fun = nn.MSELoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)  # Initialize the optimizer here
        else:
            raise ValueError("Empty layer_units. Make sure your model has at least one layer.")

    def take_step(self, X, y):
        predictions = self.model(X)
        loss = self.loss_fun(predictions, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, X, y, validation_data=None):
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        loss_df_list = []

        for epoch in range(self.max_epochs):
            for batch_features, batch_labels in dataloader:
                self.take_step(batch_features, batch_labels)

            if validation_data is not None:
                validation_loss = self.compute_loss(validation_data[0], validation_data[1])
            else:
                validation_loss = None
            loss_df_list.append(
                {'epoch': epoch, 'train_loss': self.compute_loss(X, y), 'validation_loss': validation_loss})

        self.loss_df = pd.DataFrame(loss_df_list)

    def predict(self, X):
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()

    def compute_loss(self, X, y):
        predictions = self.model(X)
        loss = self.loss_fun(predictions, y)
        return loss.item()


# Define the TorchLearnerCV class
class TorchLearnerCV:
    def __init__(self, max_epochs, batch_size, step_size, layer_units):
        self.subtrain_learner = TorchLearner(max_epochs, batch_size, step_size, layer_units)

    def fit(self, X, y, validation_data=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_epochs = 0
        best_validation_loss = float('inf')

        for max_epochs in range(1, self.subtrain_learner.max_epochs + 1):
            self.subtrain_learner.max_epochs = max_epochs
            self.subtrain_learner.fit(X_train, y_train, validation_data)

            if validation_data is not None:
                validation_loss = self.subtrain_learner.compute_loss(validation_data[0], validation_data[1])
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_epochs = max_epochs

        self.train_learner = TorchLearner(best_epochs, self.subtrain_learner.batch_size,
                                          self.subtrain_learner.step_size,
                                          [layer.out_features for layer in self.subtrain_learner.layers])
        self.train_learner.fit(X_train, y_train)

    def predict(self, X):
        return self.train_learner.predict(X)


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

    # Extract labels
    labels = dataset_df[dataset_location["output_column"]].values.reshape(-1, 1)

    # Drop non-numeric columns if specified
    if "drop_columns" in dataset_location:
        dataset_df = dataset_df.drop(columns=dataset_location["drop_columns"])

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataset_df.values)

    # Create torch tensors for features and labels
    X_tensor = torch.Tensor(scaled_features)
    y_tensor = torch.Tensor(labels)

    # Add tensors to the data_dict
    data_dict[dataset_name] = (X_tensor, y_tensor)

test_error_df_list = []

# Use KFold for train/test splits
kf = KFold(n_splits=3, shuffle=True)

test_error_df_list = []

for data_name, (X, y) in data_dict.items():
    # Define the model units for different experiments
    model_units = {
        "linear": (X.shape[1], 1),
        "deep": (X.shape[1], 100, 10, 1)
    }

    for test_fold, (train_index, test_index) in enumerate(kf.split(X)):
        for model_name, layer_units in model_units.items():
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Instantiate and fit the learner
            learner = TorchLearnerCV(max_epochs=50, batch_size=32, step_size=0.01, layer_units=layer_units)
            learner.fit(X_train, y_train, validation_data=(X_test, y_test))

            # Make predictions
            predictions = learner.predict(X_test)

            # Calculate test error
            test_error = ((predictions - y_test) ** 2).mean()

            # Append results to the test_error_df_list
            test_error_df_list.append(
                {'Dataset': data_name, 'Model': model_name, 'Fold': test_fold, 'Test Error': test_error})

# Create a DataFrame from the results
test_error_df = pd.DataFrame(test_error_df_list)

# test_error_df = pd.DataFrame(test_error_df_list)

output_file_path = "D:/home/plots/hw9.png"

# Create plots and analyze results
(ggplot(test_error_df) +
 aes(x='Fold', y='Test Error', color='Dataset + Model') +
 geom_line() +
 labs(x='Fold', y='Test Error', title='Test Error vs. Fold')
 ).save(output_file_path)
