import math
import urllib.request as download
from sklearn.metrics import accuracy_score
import plotnine as p9
import matplotlib
import os
import warnings
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
# matplotlib.use("agg")

class TorchModel(torch.nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(units_per_layer) - 1):
            self.layers.append(torch.nn.Linear(units_per_layer[i], units_per_layer[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.model = TorchModel(units_per_layer)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = torch.nn.BCEWithLogitsLoss()

    def take_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.loss_fun(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y, validation_data=None):
        subtrain_losses = []
        validation_losses = []

        for epoch in range(self.max_epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                loss = self.take_step(batch_X, batch_y)

            subtrain_loss = self.loss_fun(self.model(X), y).item()
            subtrain_losses.append(subtrain_loss)

            if validation_data:
                val_X, val_y = validation_data
                validation_loss = self.loss_fun(self.model(val_X), val_y).item()
                validation_losses.append(validation_loss)

                # print(f"Epoch {epoch + 1}/{self.max_epochs}, Subtrain Loss: {subtrain_loss:.4f}, Validation Loss: {validation_loss:.4f}")

        return subtrain_losses, validation_losses

    def decision_function(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).numpy()

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0).astype(int)


class TorchLearnerCV:
    def __init__(self, max_epochs_range, batch_size, step_size, units_per_layer):
        self.max_epochs_range = max_epochs_range
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer

    def fit(self, X, y):
        best_epochs = None
        best_validation_loss = float('inf')

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for max_epochs in self.max_epochs_range:
            subtrain_losses = []
            validation_losses = []
            for train_idx, val_idx in kf.split(X):
                subtrain_X, subtrain_y = X[train_idx], y[train_idx]
                val_X, val_y = X[val_idx], y[val_idx]

                learner = TorchLearner(max_epochs, self.batch_size, self.step_size, self.units_per_layer)
                subtrain_loss, validation_loss = learner.fit(subtrain_X, subtrain_y, validation_data=(val_X, val_y))
                subtrain_losses.extend(subtrain_loss)
                validation_losses.extend(validation_loss)

            avg_validation_loss = np.mean(validation_losses)
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                best_epochs = max_epochs

        print(f"Best number of epochs: {best_epochs}")
        return best_epochs

# data URLs
# data file paths
data_location_info = \
    {
        "spam":
            (
                "D:/home/spam.data",
                "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/spam.data"
            ),
        "zip":
            (
                "D:/home/zip.train.gz",
                "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/zip.train.gz"
            )
    }

data_info_dict = \
    {
        "spam": ("D:/home/spam.data", 57),
        "zip": ("D:/home/zip.train.gz", 0)
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
    label_col_num = int(label_col_num)  # Convert label_col_num to an integer
    data_label_vec = data_df.iloc[:, label_col_num]
    is_01 = data_label_vec.isin([0, 1])
    data01_df = data_df[is_01]  # Use boolean indexing directly on data_df
    is_label_col = data_df.columns == label_col_num
    data_features = data01_df.iloc[:, ~is_label_col]
    data_labels = data01_df.iloc[:, is_label_col]
    data_dict[data_name] = (data_features, data_labels)

spam_features, spam_labels = data_dict.pop("spam")
n_spam_rows, n_spam_features = spam_features.shape
spam_mean = spam_features.mean().to_numpy().reshape(1, n_spam_features)
spam_std = spam_features.std().to_numpy().reshape(1, n_spam_features)
spam_scaled = (spam_features - spam_mean) / spam_std
data_dict["spam_scaled"] = (spam_scaled, spam_labels)


# Initialize TorchLearnerCV with hyperparameters
max_epochs_range = [10, 20, 30]  # You can adjust this range
batch_size = 64
step_size = 0.03

linear_units_per_layer = [n_spam_features, 1]  # Linear model (no hidden layers)
deep_units_per_layer = [n_spam_features, 64, 32, 1]  # Deep neural network with 2 hidden layers

# Perform experiments
results = []

for dataset_name, (data_features, data_labels) in data_dict.items():
    if dataset_name != "zip":
        X = torch.from_numpy(data_features.to_numpy()).float()
        y = torch.from_numpy(data_labels.to_numpy()).float()

        for model_name, units_per_layer in [("TorchLinear", linear_units_per_layer),
                                            ("TorchDeep", deep_units_per_layer)]:
            learner_cv = TorchLearnerCV(max_epochs_range, batch_size, step_size, units_per_layer)
            best_epochs = learner_cv.fit(X, y)

            # Initialize lists to store losses across all folds
            all_subtrain_losses = []
            all_validation_losses = []

            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X):
                subtrain_X, subtrain_y = X[train_idx], y[train_idx]
                val_X, val_y = X[val_idx], y[val_idx]

                learner = TorchLearner(best_epochs, batch_size, step_size, units_per_layer)
                subtrain_losses, validation_losses = learner.fit(subtrain_X, subtrain_y, validation_data=(val_X, val_y))

                # Append losses from this fold to the overall lists
                all_subtrain_losses.extend(subtrain_losses)
                all_validation_losses.extend(validation_losses)

            results.append({"Dataset": dataset_name,
                            "Model": model_name,
                            "Subtrain Losses": all_subtrain_losses,
                            "Validation Losses": all_validation_losses})

# result_df = pd.DataFrame(results, columns=["Dataset", "Model", "Subtrain Losses", "Validation Losses"])
result_df = pd.DataFrame(results)
# print(len(result_df))
result_df = result_df.explode('Subtrain Losses')
result_df = result_df.explode('Validation Losses')
# print(result_df)

# Convert 'Subtrain Losses' column to float
result_df['Subtrain Losses'] = result_df['Subtrain Losses'].astype(float)
result_df['Validation Losses'] = result_df['Validation Losses'].astype(float)

from plotnine import ggplot, aes, geom_line, labs, facet_wrap, theme_minimal

plot1 = (
        ggplot(result_df, aes(x=range(len(result_df)), y="Subtrain Losses", color="Model"))
        + geom_line()
        + labs(title="Subtrain Losses", x="Epochs", y="Loss")
)

output_path = ("D:/home/plots/hw6_subtrain_losses.png")
p9.ggsave(plot1,output_path)
# print(plot1)

