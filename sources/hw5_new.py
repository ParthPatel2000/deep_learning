import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib
import os
import warnings
import pdb
warnings.filterwarnings('ignore')
matplotlib.use("agg")

# Logistic Regression with Gradient Descent
class MyLogReg:
    def __init__(self, max_iterations=1000, step_size=0.01):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.coef_ = None
        self.intercept_ = None
        self.losses = []
        
    def fit(self, X, y):
        # Convert labels to -1 and 1
        y = 2 * y - 1
        y = np.array(y).reshape(len(y))
        
        # Add a column of ones to X for the intercept
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        n_samples, n_features = X.shape
        
        self.coef_ = np.zeros(n_features)

        #pdb.set_trace()
        for iteration in range(self.max_iterations):
            pred_vec = np.dot(X, self.coef_)
            #pdb.set_trace()
            grad_loss_wrt_pred = -y / (1 + np.exp(y * pred_vec))
            gradient = np.dot(X.T, grad_loss_wrt_pred)
            self.coef_ -= self.step_size * gradient / n_samples

            # Compute and store subtrain loss
            subtrain_loss = self.compute_logistic_loss(X, y)

            # Compute and store validation loss
            pdb.set_trace()
            val_loss = self.compute_logistic_loss(X_val, y_val)

            self.losses.append({'iteration': iteration,\
                                 'set_name': 'subtrain',\
                                 'loss_value': subtrain_loss})
            
            #self.losses.append({'iteration': iteration,\
                                 #'set_name': 'validation',\
                                 #'loss_value': val_loss})

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def decision_function(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > 0, 1, 0)

    def compute_logistic_loss(self, X, y):
        pred_vec = np.dot(X, self.coef_)
        loss = np.log(1 + np.exp(-y * pred_vec))
        return np.mean(loss)

# Cross-Validation for MyLogReg
class MyLogRegCV:
    def __init__(self, max_iterations=1000, step_size=0.01, n_splits=3):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.n_splits = n_splits
        self.best_iterations = None
        self.scores_ = None
        self.lr = None

    def fit(self, X, y):
        self.scores_ = []
        best_loss = float('inf')

        # Convert labels to -1 and 1
        y = 2 * y - 1
        
        kf = KFold(n_splits=self.n_splits)
        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            lr = MyLogReg(max_iterations=self.max_iterations, step_size=self.step_size)
            lr.fit(X_train, y_train)

            train_loss = self.compute_logistic_loss(lr, X_train, y_train)
            val_loss = self.compute_logistic_loss(lr, X_val, y_val)

            self.scores_.append({'iteration': lr.max_iterations, 'set_name': 'subtrain', 'loss_value': train_loss})
            self.scores_.append({'iteration': lr.max_iterations, 'set_name': 'validation', 'loss_value': val_loss})

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_iterations = lr.max_iterations
                self.lr = lr

    def compute_logistic_loss(self, model, X, y):
        scores = model.decision_function(X)
        #pdb.set_trace()
        y = np.array(y).reshape(len(y))
        loss = np.log(1 + np.exp(-y * scores))
        return np.mean(loss)


#data URLs
spam_data_url = "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/spam.data"
zip_train_url = "https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/zip.train.gz"

#data file paths
spam_data_file_path = "D:/home/spam.data"
zip_train_file_path = "D:/home/zip.train.gz"

#checking and downloading data from urls
if not os.path.isfile(spam_data_file_path):
    download.urlretrieve(spam_data_url,spam_data_file_path)
    print('Spam File downloaded successfully')
else:
    print('Spam File already exists')

if not os.path.isfile(zip_train_file_path):
    download.urlretrieve(zip_train_url,zip_train_file_path)
    print('Zip train File downloaded successfully')
else:
    print('Zip train File already exists')


spam_pd = pd.read_csv(spam_data_file_path,sep=' ',header=None)

zip_pd = pd.read_csv(zip_train_file_path,sep=' ',header = None).dropna(axis=1)

#list for the filteration of non one zero labels
zero_one_list = [0,1]

#adjusting the zip data
zip_label_col = 0
zip_label_vec = zip_pd.iloc[:,zip_label_col]
zip_01_rows = zip_label_vec.isin(zero_one_list)
is_label_col = zip_pd.columns == zip_label_col
zip_01_df = zip_pd.loc[zip_01_rows,:]
zip_features = zip_01_df.iloc[:,~is_label_col]
zip_labels = zip_01_df.iloc[:,is_label_col]

#adjusting the spam data
spam_label_col = -1
is_spam_label_cols = spam_pd.columns == spam_label_col
spam_features = spam_pd.iloc[:,:spam_label_col]
spam_labels = spam_pd.iloc[:,spam_label_col]

#the data dict with seperate label and features for the zip and spam data
data_dict = {
    "zip" : (zip_features,zip_labels),
    "spam": (spam_features,spam_labels)
}
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize lists to store results
spam_accuracies = []
zip_accuracies = []

step_size = 0.00001
# Loop over datasets (spam and zip)
for data_name, (data_features, data_labels) in data_dict.items():
    accuracies = []

    #pdb.set_trace()
    # Loop over cross-validation folds
    for train_index, val_index in kf.split(data_features):
        X_train, X_val = data_features.iloc[train_index], data_features.iloc[val_index]
        y_train, y_val = data_labels.iloc[train_index], data_labels.iloc[val_index]

        # Initialize and fit MyLogRegCV to find the best number of iterations
        log_reg_cv = MyLogRegCV(step_size=step_size)
        log_reg_cv.fit(X_train, y_train)
        best_iterations = log_reg_cv.best_iterations

        # Initialize and fit MyLogReg with the best number of iterations
        log_reg = MyLogReg(max_iterations=best_iterations, step_size=step_size)
        log_reg.fit(X_train, y_train)

        # Make predictions on the validation set
        val_predictions = log_reg.predict(X_val)

        # Calculate accuracy and store it
        accuracy = accuracy_score(y_val, val_predictions)
        accuracies.append(accuracy)

    # Calculate the mean accuracy for this dataset
    mean_accuracy = np.mean(accuracies)

    # Print the mean accuracy
    print(f"Mean Accuracy for {data_name}: {mean_accuracy}")

    # Store the mean accuracy in the appropriate list
    if data_name == "spam":
        spam_accuracies.append(mean_accuracy)
    elif data_name == "zip":
        zip_accuracies.append(mean_accuracy)

# Visualize the results
accuracy_df = pd.DataFrame({
    'Dataset': ["Spam", "ZIP"],
    'Mean Accuracy': spam_accuracies + zip_accuracies
})

scores_df = pd.DataFrame(log_reg_cv.scores_)
output_file_path = r"D:\home\plots\hw5_loss.png"
myplot = (
   p9.ggplot(scores_df)
    +p9.aes(x="iteration",y="loss_value")
    +p9.geom_line()
)
p9.ggsave(myplot,output_file_path)
