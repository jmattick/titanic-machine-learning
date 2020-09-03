# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# input file path
file_path = 'train.csv'

# read data using pandas
data = pd.read_csv(file_path)

# drop missing values
data = data.dropna(axis=0)

# set target
y = data.Survived

# select features
#features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
features = ['Pclass','Age','Fare']

# subset data
X = data[features]

# split data into training and validation datasets
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

# define model
decision_tree_model = DecisionTreeRegressor(random_state=1)

# Fit model
decision_tree_model.fit(train_X, train_y)

# get predicted values
val_predict = decision_tree_model.predict(val_X)

# print mean absolute error
print("mean absolute error of prediction: " + str(mean_absolute_error(val_y, val_predict)))
