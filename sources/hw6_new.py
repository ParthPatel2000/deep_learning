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
matplotlib.use("agg")

class TorchModel(nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()
        layers = []
        for i in range(len(units_per_layer) - 1):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i + 1]))
            if i < len(units_per_layer) - 2:
                layers.append(nn.ReLU())
        self.stack = nn.Sequential(*layers)

    def forward(self, features):
        return self.stack(features)

class TorchLearner:
    def __init__(self, units_per_layer, max_epochs, batch_size, step_size):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size

        self.model = TorchModel(units_per_layer)
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.BCEWithLogitsLoss()

    def take_step(self, X, y):
        self.optimizer.zero_grad()
        predictions = self.model(X)
        loss = self.loss_fun(predictions, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X_train, y_train, X_val, y_val):
        train_losses = []
        val_losses = []

        for epoch in range(self.max_epochs):
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                loss = self.take_step(X_batch, y_batch)

            train_loss = self.loss_fun(self.model(X_train), y_train).item()
            val_loss = self.loss_fun(self.model(X_val), y_val).item()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.max_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return train_losses, val_losses

    def decision_function(self, X):
        with torch.no_grad():
            return torch.sigmoid(self.model(X)).numpy()

    def predict(self, X):
        probabilities = self.decision_function(X)
        return (probabilities > 0.5).astype(int)

class TorchLearnerCV:
    def __init__(self, units_per_layer, max_epochs, batch_size, step_size, n_folds=3):
        self.units_per_layer = units_per_layer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.n_folds = n_folds

    def fit(self, X, y):
        kf = KFold(n_splits=self.n_folds)

        best_epochs = []
        best_val_losses = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = TorchLearner(self.units_per_layer, self.max_epochs, self.batch_size, self.step_size)
            train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val)

            best_epoch = np.argmin(val_losses)
            best_epochs.append(best_epoch + 1)  # Add 1 because epochs are 1-indexed
            best_val_losses.append(val_losses[best_epoch])

        best_avg_epoch = int(np.mean(best_epochs))
        return best_avg_epoch, best_val_losses



# class TorchModel(torch.nn.Module):
#     def _init_(self, units_per_layer):
#         super(TorchModel, self)._init_()
#         seq_args = []
#         second_to_last = len(units_per_layer)-1
#         for layer_i in range(len(units_per_layer)-1):
#             next_i = layer_i + 1
#             layer_units = units_per_layer[layer_i]
#             next_units = units_per_layer[next_i]
#             seq_args.append(torch.nn.Linear(layer_units, next_units))
#             if layer_i < second_to_last-1:
#                 seq_args.append(torch.nn.RELU())
#         self.stack = torch.nn.Sequential(*seq_args)
#         #self.weight_vec = torch.nn.Linear(n_features, 1)
#     def forward(self, features):
#         return self.stack(features)

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

# for dataset_name, (data_features, data_labels) in data_dict.items():
#     if dataset_name != "zip":
#         X = torch.from_numpy(data_features.to_numpy()).float()
#         y = torch.from_numpy(data_labels.to_numpy()).float()

#         for model_name, units_per_layer in [("TorchLinear", linear_units_per_layer), ("TorchDeep", deep_units_per_layer)]:
#             learner_cv = TorchLearnerCV(max_epochs_range, batch_size, step_size, units_per_layer)
#             best_epochs = learner_cv.fit(X, y)

#             learner = TorchLearner(best_epochs, batch_size, step_size, units_per_layer)
#             subtrain_losses, validation_losses = learner.fit(X, y)
#             results.append((dataset_name, model_name, subtrain_losses, validation_losses))

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

            results.append((dataset_name, model_name, all_subtrain_losses, all_validation_losses))

result_df = pd.DataFrame(results, columns=["Dataset", "Model", "Subtrain Losses", "Validation Losses"])

from plotnine import ggplot, aes, geom_line, labs, facet_wrap, theme_minimal

plot1 = (
        ggplot(result_df, aes(x=range(len(result_df)), y="Subtrain Losses", color="Model"))
        + geom_line()
        + labs(title="Subtrain Losses", x="Epochs", y="Loss")
        + theme_minimal()
)

print(plot1)

# import pdb
#
# # Plot subtrain and validation losses
# pdb.set_trace()
# for dataset_name, model_name, subtrain_losses, validation_losses in results:
#     if dataset_name != "zip":
#         plt.figure()
#         plt.plot(subtrain_losses, label="Subtrain Loss")
#         plt.plot(validation_losses, label="Validation Loss")
#         plt.title(f"{model_name} on {dataset_name}")
#         plt.xlabel("Epochs")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.show()

# Now, perform experiments/application using other models (KNeighborsClassifier, LogisticRegression, etc.) and compare test accuracies.

# Define and train
