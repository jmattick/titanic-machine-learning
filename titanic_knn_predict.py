# import libraries
import sys, getopt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# get arguments
args = sys.argv[1:]

# get parameters
train_file = 'train.csv'
test_file = 'test.csv'
output_file = 'predictions.csv'
try:
    opts, args = getopt.getopt(args, "ht:i:o:", ["train=", "test=", "output="])
except getopt.GetoptError:
    print('test.py -i <inputfile> -o <outputfile>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-t", "--train"):
        train_file = arg
    elif opt in ("-i", "--test"):
        test_file = arg
    elif opt in ("-o", "--output"):
        output_file = arg

# indicate input files
print('Training dataset: ' + str(train_file))
print('Test dataset: ' + str(test_file))

# read data using pandas
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# select features
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'Parch']


def select_features_split(data, y, features):
    """Function to subset features and split training and validation sets.
    Scales Data"""
    # subset data
    X = data[features]

    # split data into training and validation datasets
    train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

    return train_X, val_X, train_y, val_y


def scale_data(train_X, val_X):
    """Function scales data using StandardScaler"""
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
    
    return label_train_X, label_val_X


def test_model(train_X, val_X, train_y, val_y, model):
    """Function to return the accuracy of a given model and dataset"""
    # Fit model
    model.fit(train_X, train_y)

    # get predicted values
    val_predict = model.predict(val_X)

    # return accuracy
    return accuracy_score(val_y, val_predict)


def run_knn_prediction(train, val, y, features, k):
    """Function to predict survial using a knn model given a training and
    test dataset"""
    # setup list to subset data
    first_subset = []
    first_subset.extend(features)
    first_subset.append(y)

    # subset data
    train_X = train[first_subset]  # include passenger id and survived
    val_X = val[features]

    # drop missing values
    train_X = train_X.dropna(axis=0)
    val_X = val_X.dropna(axis=0)

    # separate passenger ID and y
    train_ID = train_X['PassengerId']
    train_y = train_X[y]
    val_ID = val_X['PassengerId']

    # remove passenger ID in train and val dataset
    train_X = train_X.drop(['PassengerId', 'Survived'], axis=1)
    val_X = val_X.drop(['PassengerId'], axis=1)

    # label encoding
    s = (train_X.dtypes == 'object')
    # if columns need to be encoded
    if len(s) > 0:
        # select features with object dtype
        features_to_encode = list(s[s].index)
        # encode categorical columns using label encoding
        label_train_X, label_val_X = label_encoding(train_X, val_X, features_to_encode)
    else:
        label_train_X, label_val_X = train_X, val_X

    # scale
    label_train_X, label_val_X = scale_data(label_train_X, label_val_X)

    # define model
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit model
    knn.fit(label_train_X, train_y)

    # get predicted values
    val_predict = knn.predict(label_val_X)

    return val_ID, val_predict


# run model
res = run_knn_prediction(train_data, test_data, 'Survived', features, 5)

# extract ids and predicted values
id = list(res[0])
predicted = list(res[1])

# output results
with open(output_file, 'w') as z:
    z.write('Passenger Id, Predicted Survival\n')
    for i in range(len(id)):
        z.write(str(id[i]) + ',' + str(predicted[i]) + '\n')

print('Results output to: ' + str(output_file))
