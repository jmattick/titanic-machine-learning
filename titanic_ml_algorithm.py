# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from itertools import combinations

# input file path
file_path = 'train.csv'

# read data using pandas
data = pd.read_csv(file_path)

# drop missing values
data = data.dropna(axis=0)

# set target
y = data.Survived

# select features
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

# subset data
X = data[features]

# get list of categorical variables
s = (X.dtypes == 'object')
features_to_encode = list(s[s].index)

# split data into training and validation datasets
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

def select_features_split(data, y, features):
    """Function to subset features and split training and validation sets.
    Scales Data"""
    # subset data
    X = data[features]

    # split data into training and validation datasets
    train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

    return train_X, val_X, train_y, val_y

def drop_cols(train_X, val_X, cols):
    """Function to drop columns"""
    # copy data
    drop_train_X = train_X.copy()
    drop_val_X = val_X.copy()

    # drop columns
    drop_train_X = drop_train_X.drop(cols, axis=1)
    drop_val_X = drop_val_X.drop(cols, axis =1)
    
    return drop_train_X, drop_val_X

def scale_data(train_X, val_X):
    # scale data
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    return train_X, val_X

def label_encoding(train_X, val_X, cols):
    """Function using label encoding to convert categorical columns
    to numbers"""
    # copy data
    label_train_X = train_X.copy()
    label_val_X = val_X.copy()

    # define label encoder
    label_encoder = LabelEncoder()

    # encode labels for each feature in list
    for col in cols:
        label_train_X[col] = label_encoder.fit_transform(train_X[col])
        label_val_X[col] = label_encoder.transform(val_X[col])

    # scale
    label_train_X, label_val_X = scale_data(label_train_X, label_val_X)
    
    return label_train_X, label_val_X

def oh_encoding(train_X, val_X, cols):
    """Function using one-hot encoding to convert categorical columns
    to numbers"""
    # copy data
    oh_train_X = train_X.copy()
    oh_val_X = val_X.copy()

    # define one-hot encoder
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # encode labels for each feature 
    oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(train_X[cols]))
    oh_cols_val = pd.DataFrame(oh_encoder.transform(val_X[cols]))

    # re-add index
    oh_cols_train.index = train_X.index
    oh_cols_val.index = val_X.index

    # remove old columns
    oh_train_X = oh_train_X.drop(cols, axis=1)
    oh_val_X = oh_val_X.drop(cols, axis=1)

    # add oh columns
    oh_train_X = pd.concat([oh_train_X, oh_cols_train], axis=1)
    oh_val_X = pd.concat([oh_val_X, oh_cols_val], axis=1)

    # scale
    oh_train_X, oh_val_X = scale_data(oh_train_X, oh_val_X)
    
    return oh_train_X, oh_val_X

def test_model(train_X, val_X, train_y, val_y, model):
    """Function to return the mean absolute error of a given model and dataset"""
    # Fit model
    model.fit(train_X, train_y)

    # get predicted values
    val_predict = model.predict(val_X)

    # return mean absolute error
    return accuracy_score(val_y, val_predict)

def test_features(data, y, features, model):
    """Function to test all combinations of features using both label encoding
    and on-hot encoding given a model"""
    label_encode_list = np.array([])
    oh_encode_list = np.array([])

    # subset data
    for feature in features:
        X = data[feature]
        # subset features and split training and validation datasets
        train_X, val_X, train_y, val_y = select_features_split(data, y, feature)
        # get list of categorical variables
        s = (X.dtypes == 'object')
        # if columns need to be encoded
        if len(s) > 0:
            #select features with object dtype
            features_to_encode = list(s[s].index)
            # encode categorical columns using one-hot or label encoding
            oh_train_X, oh_val_X = oh_encoding(train_X, val_X, features_to_encode)
            label_train_X, label_val_X = label_encoding(train_X, val_X, features_to_encode)
        else:
            oh_train_X, oh_val_X = train_X, val_X
            label_train_X, label_val_X = train_X, val_X
        # calculate mean absolute error for both types of encoding
        oh_mae = test_model(oh_train_X, oh_val_X, train_y, val_y, model)
        label_mae = test_model(label_train_X, label_val_X, train_y, val_y, model)
        # add mae to np arrays
        oh_encode_list = np.append(oh_encode_list, oh_mae)
        label_encode_list = np.append(label_encode_list, label_mae)

    # find min mean absolute error
    if oh_encode_list.min() < label_encode_list.min():
        encoding_method = "oh_encoding"
        best_feature_list = features[np.argmax(oh_encode_list)]
        mae = oh_encode_list.max()
    else:
        encoding_method = "label_encoding"
        best_feature_list = features[np.argmax(label_encode_list)]
        mae = oh_encode_list.max()
    return encoding_method, best_feature_list, mae

    

#drop categorical variables
drop_train_X, drop_val_X = drop_cols(train_X, val_X, features_to_encode)

# apply label encoding to non-numerical features
label_train_X, label_val_X = label_encoding(train_X, val_X, features_to_encode)

# apply one-hot encoding to non-numerical features
oh_train_X, oh_val_X = oh_encoding(train_X, val_X, features_to_encode)

# define model
logistic_regression_model = LogisticRegression(random_state=1)

# test model
print('Testing categorical encoding: ')
print('drop categorical: ' + str(test_model(drop_train_X, drop_val_X, train_y, val_y, logistic_regression_model)))
print('label encode categorical: ' + str(test_model(label_train_X, label_val_X, train_y, val_y, logistic_regression_model)))
print('one-hot encode categorical: ' + str(test_model(oh_train_X, oh_val_X, train_y, val_y, logistic_regression_model)))

print('Testing features: ')

# get all combinations of features
feature_combinations = sum([list(map(list, combinations(features, i))) for i in range(len(features) + 1)], [])
feature_combinations = feature_combinations[1:]
print(test_features(data, y, feature_combinations, logistic_regression_model))
