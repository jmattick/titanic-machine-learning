# titanic-machine-learning
Jessica Mattick

## Info
Dataset from: https://www.kaggle.com/c/titanic

The dataset train.csv was used to train a machine 
learning model to predict the survival of passengers. The data was 
analyzed using pandas, numpy, and sklearn. 
The code used to determine the parameters of the model is located
in `titanic_ml_algorithm.py`.

After reading the dataset using pandas, any rows with na values 
were dropped.
```
import pandas as pd

# read data using pandas
data = pd.read_csv(file_path)

# drop missing values
data = data.dropna(axis=0)
``` 

The data is summarized below:


| Column      | Non-Null Count | Dtype
| ------      | -------------- | -----
| PassengerId | 183 non-null   | int64
| Survived    | 183 non-null   | int64
| Pclass      | 183 non-null   | int64
| Name        | 183 non-null   | object
| Sex         | 183 non-null   | object
| Age         | 183 non-null   | float64
| SibSp       | 183 non-null   | int64
| Parch       | 183 non-null   | int64
| Ticket      | 183 non-null   | object
| Fare        | 183 non-null   | float64
| Cabin       | 183 non-null   | object
| Embarked    | 183 non-null   | object

The Survived column was set as the target.
```
y = data.Survived
```
The features most likely to make an impact on survival were 
selected. Features such as Name or Passenger Id were excluded. 

```
# select features
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
```
The dataset was subset by features and split into training and test
datasets using the sklearn train_test_split method.

```
from sklearn.model_selection import train_test_split
def select_features_split(data, y, features):
    """Function to subset features and split training and validation sets.
    Scales Data"""
    # subset data
    X = data[features]

    # split data into training and validation datasets
    train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

    return train_X, val_X, train_y, val_y
```
Since there are columns in the dataset that are not numeric,
they need to be encoded. Two encoding methods were tested. 

(1) Label encoding:

Example | Example Encoded
----    | ----
group1  | 0
group2  | 1
group1  | 0

(2) One-hot encoding:

Example | group1    | group2
----    | ----      | ----
group1  | 1 | 0
group2  | 0 | 1
group1  | 1 | 0

Functions were generated for each encoding method using sklearn. 

```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
    
    return oh_train_X, oh_val_X
```
To evaluate the models, the accuracy was compared. This
was calculated using the sklearn accuracy_score method.
A function was created to fit/predict using the given model
and return the accuracy. 

```
from sklearn.metrics import accuracy_score

def test_model(train_X, val_X, train_y, val_y, model):
    """Function to return the accuracy of a given model and dataset"""
    # Fit model
    model.fit(train_X, train_y)

    # get predicted values
    val_predict = model.predict(val_X)

    # return accuracy
    return accuracy_score(val_y, val_predict)
```

Data was scaled using the sklearn StandardScaler. 
A function was written to scale the data for both the 
training set and the test set.

```
from sklearn.preprocessing import StandardScaler

def scale_data(train_X, val_X):
    """Function scales data using StandardScaler"""
    # scale data
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    return train_X, val_X
```

All possible combinations of the selected features
were calculated using itertools combinations method.

```
from itertools import combinations

# get all combinations of features
feature_combinations = sum([list(map(list, combinations(features, i))) for i in range(len(features) + 1)], [])
feature_combinations = feature_combinations[1:]
```

A function was written to compare the accuracy of 
each feature combination using label encoding or 
one-hot encoding. The feature combination and encoding 
method with the highest accuracy is returned. 

```
f test_features(data, y, features, model):
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
            # select features with object dtype
            features_to_encode = list(s[s].index)
            # encode categorical columns using one-hot or label encoding
            oh_train_X, oh_val_X = oh_encoding(train_X, val_X, features_to_encode)
            label_train_X, label_val_X = label_encoding(train_X, val_X, features_to_encode)
        else:
            oh_train_X, oh_val_X = train_X, val_X
            label_train_X, label_val_X = train_X, val_X
        # scale
        oh_train_X, oh_val_X = scale_data(oh_train_X, oh_val_X)
        label_train_X, label_val_X = scale_data(label_train_X, label_val_X)
        # calculate accuracy for both types of encoding
        oh_acc = test_model(oh_train_X, oh_val_X, train_y, val_y, model)
        label_acc = test_model(label_train_X, label_val_X, train_y, val_y, model)
        # add accuracy to np arrays
        oh_encode_list = np.append(oh_encode_list, oh_acc)
        label_encode_list = np.append(label_encode_list, label_acc)

    # find max accuracy
    if oh_encode_list.min() < label_encode_list.min():
        encoding_method = "oh_encoding"
        best_feature_list = features[np.argmax(oh_encode_list)]
        acc = oh_encode_list.max()
    else:
        encoding_method = "label_encoding"
        best_feature_list = features[np.argmax(label_encode_list)]
        acc = oh_encode_list.max()
    return encoding_method, best_feature_list, acc
```

Two models were tested across all feature combinations and 
encoding methods. The first model tested was 
LogisticRegression from sklearn. The test_features function
was called with Logistic Regression model.

```
from sklearn.linear_model import LogisticRegression
# define model
logistic_regression_model = LogisticRegression(random_state=1)
# test features across model
test_features(data, y, feature_combinations, logistic_regression_model)
```

The results of test_features using a Logistic Regression model determined
the following:

best encoding method: label encoding
best features: Sex, Age, Parch, Embarked
accuracy: 0.76

The second model tested was the k-nearest neighbors model (KNN).
The test_features function was called using
the KNeighborsClassifier model from sklearn. This model was
tested across 3 different values of k (3,5,10).

```
from sklearn.neighbors import KNeighborsClassifier

k_values = [3,5,10]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    test_features(data, y, feature_combinations, knn)
```

The results of this testing determined the following:

best encoding method: label encoding
best features: Pclass, Sex, Age, Parch
best k: 5
accuracy: 0.83

The KNN method using k=5 and the features Pclass, Sex, 
Age, Parch had the highest accuracy out of all the 
methods tested. This method was chosen to predict the 
survival of passengers. 

The `titanic_knn_predict.py` script applies the KNN method to predict the 
survival of passengers using the parameters determined by running 
the `titanic_ml_algorithm.py` script. The `titanic_knn_predict.py` script
predicts survival using the 'Pclass', 'Sex', 'Age', 'Parch' features with 
k=5. The `titanic_knn_predict.py` script requires a training dataset in 
addition to the input dataset to be predicted. The output is a csv file 
containing the Passenger Ids and predicted survival. The accuracy of this 
model determined using the provided train.csv file is 0.83. 
 
## Usage

Install dependencies using provided environment file: 
```
conda env create -f environment.yml
```

Predict survival using `titanic_knn_predict.py`. 

Two input csv files are required:

1. A training dataset containing the features and survival information

2.  A test dataset containing features to be used for prediction

Example files are included in the repository from https://www.kaggle.com/c/titanic.

Parameters: 

- `-t` or `--train`: training csv file
- `i` or `--test`: test csv file for predictions
- `-o` or `--output`: output csv file

Example: 
```
python titanic_knn_predict.py -t train.csv -i test.csv -o predictions.csv
```
