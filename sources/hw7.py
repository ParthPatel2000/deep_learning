import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
import urllib.request as download
import plotnine as p9
from sklearn.impute import SimpleImputer


# Classes
class InitialNode:
    """Leaf node in computation graph (not derived from other nodes)"""

    def __init__(self, value):
        """save value as attribute"""
        self.value = value
        self.grad = None


class Operation:
    def __init__(self, *node_list):
        """store nodes in list as attributes with names given in input_names"""
        self.input_nodes = node_list
        self.value = None

    def backward(self):
        """call self.gradient, then save results in grad attribute of parent
        nodes, then call backward() on parents if necessary."""
        grad_values = self.gradient()
        for node, grad_value in zip(self.input_nodes, grad_values):
            node.grad = grad_value
            if isinstance(node, Operation):
                node.backward()


class mm(Operation):
    """Matrix multiply"""
    input_names = ('A', 'B')

    def forward(self):
        self.value = np.dot(self.input_nodes[0].value, self.input_nodes[1].value)

    def gradient(self):
        return (np.dot(self.input_nodes[1].value.T, self.input_nodes[0].grad),
                np.dot(self.input_nodes[0].value.T, self.input_nodes[1].grad))


class relu(Operation):
    """Non-linear activation"""
    input_names = ('X',)

    def forward(self):
        self.value = np.maximum(0, self.input_nodes[0].value)

    def gradient(self):
        return (np.where(self.input_nodes[0].value > 0, self.input_nodes[0].grad, 0),)


class logistic_loss(Operation):
    """Loss of predicted scores given true labels"""
    input_names = ('Y_pred', 'Y_true')

    def forward(self):
        scores = self.input_nodes[0].value
        labels = self.input_nodes[1].value
        self.value = -np.mean(labels * np.log(sigmoid(scores)) + (1 - labels) * np.log(1 - sigmoid(scores)))

    def gradient(self):
        return (self.input_nodes[0].value - self.input_nodes[1].value,)  # gradient of logistic loss


# Helper function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# AutoMLP class


class AutoMLP:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer, intercept=False):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.intercept = intercept

        # Initialize weight_node_list
        self.weight_node_list = []
        for i in range(len(units_per_layer) - 1):
            input_size = units_per_layer[i] + int(intercept)  # Add 1 if intercept is True
            output_size = units_per_layer[i + 1]
            weight_matrix = np.random.randn(input_size, output_size) * 0.01  # Initialize with small random values
            weight_node = InitialNode(weight_matrix)
            self.weight_node_list.append(weight_node)

    def get_pred_node(self, X):
        # Initialize input node
        input_node = InitialNode(X)

        # Forward pass through the layers
        pred_node = input_node
        for i, weight_node in enumerate(self.weight_node_list):
            pred_node = mm(pred_node, weight_node)
            if i < len(self.weight_node_list) - 1:  # Apply relu only for hidden layers
                pred_node = relu(pred_node)

        return pred_node

    def take_step(self, X, y):
        # Make an InitialNode instance from y
        label_node = InitialNode(y)

        # Use get_pred_node(X) to get a node for predicted scores
        pred_node = self.get_pred_node(X)

        # Instantiate the last node in your computation graph
        loss_node = logistic_loss(pred_node, label_node)

        # Call backward() on the final node instance (mean loss) to compute and store gradients in each node
        loss_node.backward()

        # Update each parameter matrix (take a step in the negative gradient direction)
        for weight_node in self.weight_node_list:
            weight_node.value -= self.step_size * weight_node.grad

    def fit(self, X, y):
        # Gradient descent learning of weights
        dl = DataLoader(X, y, batch_size=self.batch_size)
        loss_df_list = []
        for epoch in range(self.max_epochs):
            for batch_features, batch_labels in dl:
                self.take_step(batch_features, batch_labels)
            # TODO: Compute subtrain/validation loss using current weights and append to loss_df_list
        self.loss_df = pd.concat(loss_df_list)

    def decision_function(self, X):
        # Return numpy vector of predicted scores
        pred_node = self.get_pred_node(X)
        return pred_node.value

    def predict(self, X):
        # Return numpy vector of predicted classes
        scores = self.decision_function(X)
        return np.where(scores > 0.5, 1, 0)


# DataLoader class for batch-wise processing

class DataLoader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = len(X)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        batch_features = self.X[self.current_index:self.current_index + self.batch_size]
        batch_labels = self.y[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch_features, batch_labels


# AutoGradLearnerCV class

class AutoGradLearnerCV:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer

        self.subtrain_model = AutoMLP(max_epochs=max_epochs,
                                      batch_size=batch_size,
                                      step_size=step_size,
                                      units_per_layer=units_per_layer)

    def fit(self, X, y):
        # Cross-validation for selecting the best number of epochs
        # TODO: Implement cross-validation and find the best number of epochs
        best_epochs = 10  # Placeholder for demonstration
        self.subtrain_model.fit(X, y)

        # Train the model on the entire dataset using the best number of epochs
        self.train_model = AutoMLP(max_epochs=best_epochs,
                                   batch_size=self.batch_size,
                                   step_size=self.step_size,
                                   units_per_layer=units_per_layer)
        self.train_model.fit(X, y)

    def predict(self, X):
        return self.train_model.predict(X)


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
    data_df = True

    if data_name == "zip":
        data_df = pd.read_csv(data_info_dict["zip"][0], sep=' ', header=None, compression="gzip")
        imputer = SimpleImputer(strategy='median')
        data_df = pd.DataFrame(imputer.fit_transform(data_df))
    else:
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

# Experiments/application

for data_name, (data_scaled, data_labels) in data_dict.items():
    data_dict[data_name] = {"X": data_scaled, "y": data_labels}

test_acc_df_list = []

for data_name, data in data_dict.items():
    model_units = \
        {
            "linear": (data["X"].shape[1], 1),
            "deep": (data["X"].shape[1], 100, 10, 1)
        }
    for test_fold, indices in enumerate(KFold(n_splits=3).split(data["X"])):
        for model_name, units_per_layer in model_units.items():
            # TODO: Fit(train data), then predict(test data), then store accuracy
            test_row = {"data_name": data_name, "test_fold": test_fold, "model_name": model_name, "accuracy": 0.8}
            test_acc_df_list.append(test_row)

test_acc_df = pd.DataFrame(test_acc_df_list)

print(test_acc_df)

# Plotting with white background
plot = (p9.ggplot(test_acc_df, p9.aes(x='test_fold', y='accuracy', color='model_name'))
        + p9.facet_wrap('~data_name')
        + p9.geom_line()
        + p9.theme(panel_background=p9.element_rect(fill='white'))  # Add white background to the panel
        + p9.geom_point()  # Add points to the plot
        + p9.geom_text(p9.aes(label='accuracy'), nudge_y=0.02, color='black', size=8))  # Add text labels for accuracy

output_file_path = r"d:\home\plots\hw7.png"
plot.save(output_file_path)
