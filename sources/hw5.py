import urllib.request as download
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import plotnine as p9
import matplotlib
import os
import pandas as pd
import numpy as np
import warnings
import pdb
warnings.filterwarnings('ignore')
matplotlib.use("agg")

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

for data_name,(data_features,data_labels) in data_dict.items():

    data_nrow,data_ncol = data_features.shape
    weight_vec = np.repeat(0.0,data_ncol).reshape(data_ncol,1)
    data_mat = data_features.to_numpy()

    for iteration in range(10):
        pdb.set_trace()
        pred_vec = np.matmul(data_features,weight_vec)
        label_pos_neg_vec = np.where(data_labels==1,1,-1)
        grad_loss_wrt_pred = label_pos_neg_vec/(1+np.exp(label_pos_neg_vec*pred_vec))
        loss_vec = np.log(1+np.exp(-label_pos_neg_vec*pred_vec))
        grad_loss_wrt_weight = np.matmul(data_mat.T ,grad_loss_wrt_pred)
        step_size = 0.001
        weight_vec -= step_size * grad_loss_wrt_pred 
