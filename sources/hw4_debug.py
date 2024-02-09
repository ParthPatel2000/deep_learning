import urllib.request
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


class MyKNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

    def predict(self, test_features):
        predictions = []
        for test_sample in test_features:
            distances = np.sqrt(np.sum((self.train_features - test_sample) ** 2, axis=1))
            nearest_neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbor_labels = self.train_labels[nearest_neighbor_indices]

            '''using bincount to count the occurrences of the labels in the array
            and storing them in an array of counts where each index has the
            number of occurrence of that index(number) in the labels array stored in it.
            then using argmax to find the index of the maximum number from the predicted_label.'''
            predicted_label = np.bincount(nearest_neighbor_labels).argmax()
            predictions.append(predicted_label)
        return np.array(predictions)

class MyCV:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = {}

    def fit_one(self, param_dict, train_features, train_labels):
        self.estimator.n_neighbors = param_dict['n_neighbors']
        self.estimator.fit(train_features, train_labels)

    def fit(self, train_features, train_labels):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        validation_df_list = []

        # Check that train_features and train_labels have the same number of samples
        if train_features.shape[0] != train_labels.shape[0]:
            raise ValueError("Number of samples in train_features and train_labels must match.")

        
        for 
            pdb.set_trace()
            print(sub_train_indices)
            print(train_features[sub_train_indices].shape, train_labels[sub_train_indices].shape)
            subtrain_data = {
                "subtrain": (train_features[sub_train_indices], train_labels[sub_train_indices]),
                "validation_fold": validation_fold
            }

            for param_dict in self.param_grid:
                self.fit_one(param_dict, **subtrain_data)
                y_pred = self.estimator.predict(train_features[val_indices])
                accuracy = accuracy_score(train_labels[val_indices], y_pred)
                validation_row = {
                    "validation_fold": validation_fold,
                    "accuracy_percent": accuracy * 100,
                    **param_dict
                }
                validation_df_list.append(validation_row)

        validation_df = pd.DataFrame(validation_df_list)
        best_param_dict = validation_df.groupby('n_neighbors')['accuracy_percent'].mean().idxmax()
        self.best_params_ = {"n_neighbors": best_param_dict}
        self.fit_one(self.best_params_, train_features, train_labels)

    def predict(self, test_features):
        return self.estimator.predict(test_features)


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

#defining the number of splits for K nearest neighbors
splits = 3

accuracy_list = []

pred_dict ={
    "zip":{"nearest_neighbor":[],"linear_model":[],"featureless":[]},
    "spam":{"nearest_neighbor":[],"linear_model":[],"featureless":[]}
}
for data_set ,(data_features,data_labels) in data_dict.items():
    seed = 472
    print("DataSet: ",data_set)

    #setting up the KFOLD method for splitting the 
    kf_object = KFold(n_splits=splits,shuffle=True,random_state = seed)
    
    #splitting the data for kfold evaluation,
    #split will divide the data into the K number of sets as defined earlier 
    enum_obj = enumerate(kf_object.split(data_features))

    for fold_number, index_tuple in enum_obj:
        print("Fold number: ", fold_number)
        zip_obj = zip(["train","test"],index_tuple)
        split_data_dict = {}

    
        for set_name,set_indices in zip_obj:
            split_data_dict[set_name]=(
                data_features.iloc[set_indices,:],
                data_labels.iloc[set_indices]
            )
        #fitting the data into the KNN using
        max_neighbors = 20
        '''clf = GridSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors":[k+1 for k in range(max_neighbors)]})'''
        clf = MyCV(estimator=MyKNN(1),\
                         param_grid=[{'n_neighbors':n_neighbors} for n_neighbors in range(max_neighbors)],cv=5)
        #training
        train_features,train_labels = split_data_dict["train"]
        clf.fit(data_features,data_labels)

        best_neighbors_count = clf.best_params_
        most_freq_label = train_labels.value_counts().index[0]

        #testing
        test_features,test_labels=split_data_dict["test"]
        pred_labels_knn = clf.predict(test_features)

        #extend prediction lists correctly
        pred_dict[data_set]["nearest_neighbor"].extend(pred_labels_knn)

        

        #accuracy list
        accuracy_list.append({"data_set": data_set,
                                   "fold_id": fold_number,
                                   "algorithm": "Nearest Neighbors",
                                   "test_accuracy_percent": accuracy_knn})


accuracy_dataframe = pd.DataFrame(accuracy_list)
print(accuracy_dataframe)

plot_save_path = "d:/home/hw3_plot.png"
plot = p9.ggplot()+\
    p9.geom_point(
        p9.aes(
            x="test_accuracy_percent",
            y="algorithm"
        ),
        data = accuracy_dataframe
    )+\
    p9.facet_grid(".~data_set")

plot.save(plot_save_path)
   
    
