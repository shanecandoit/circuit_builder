
import numpy as np
import pandas as pd

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

print(X.shape, y.shape)
print("X", X[0])
print("Y", y[0])

iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['target'] = y
iris_df['target'] = iris_df['target'].astype('category')

print(iris_df.head())
"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) target
0                5.1               3.5                1.4               0.2      0
1                4.9               3.0                1.4               0.2      0
2                4.7               3.2                1.3               0.2      0
3                4.6               3.1                1.5               0.2      0
4                5.0               3.6                1.4               0.2      0
"""

# describe the dataset
print(iris_df.describe())
"""

       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000
mean            5.843333          3.057333           3.758000          1.199333
std             0.828066          0.435866           1.765298          0.762238
min             4.300000          2.000000           1.000000          0.100000
25%             5.100000          2.800000           1.600000          0.300000
50%             5.800000          3.000000           4.350000          1.300000
75%             6.400000          3.300000           5.100000          1.800000
max             7.900000          4.400000           6.900000          2.500000
"""

iris_df.to_csv('iris.csv', index=False)

# One-hot encode the target
iris_df = pd.get_dummies(iris_df, columns=['target'], prefix='target', dtype=int)
# iris_df.to_csv('datasets/iris_onehot.csv', index=False)

# make a new dataframe with numbers in quartiles, so its binary
iris_df_binary = iris_df.copy()
print(iris_df_binary.head())

SepLen_cuts = [5.1, 5.8, 6.4]
iris_df_binary['SepLen_1'] = (iris_df_binary['sepal length (cm)'] < SepLen_cuts[0]) # to int
iris_df_binary['SepLen_1'] = iris_df_binary['SepLen_1'].astype(int)
iris_df_binary['SepLen_2'] = (iris_df_binary['sepal length (cm)'] >= SepLen_cuts[0]) & (iris_df_binary['sepal length (cm)'] < SepLen_cuts[1])
iris_df_binary['SepLen_2'] = iris_df_binary['SepLen_2'].astype(int)
iris_df_binary['SepLen_3'] = (iris_df_binary['sepal length (cm)'] >= SepLen_cuts[1]) & (iris_df_binary['sepal length (cm)'] < SepLen_cuts[2])
iris_df_binary['SepLen_3'] = iris_df_binary['SepLen_3'].astype(int)
iris_df_binary['SepLen_4'] = (iris_df_binary['sepal length (cm)'] >= SepLen_cuts[2])
iris_df_binary['SepLen_4'] = iris_df_binary['SepLen_4'].astype(int)
# drop the original column
iris_df_binary.drop(columns=['sepal length (cm)'], inplace=True)

# SepWid_cuts = [2.8, 3.0, 3.3]
SepWid_cuts = [2.8, 3.0, 3.3]
iris_df_binary['SepWid_1'] = (iris_df_binary['sepal width (cm)'] < SepWid_cuts[0]) # to int
iris_df_binary['SepWid_1'] = iris_df_binary['SepWid_1'].astype(int)
iris_df_binary['SepWid_2'] = (iris_df_binary['sepal width (cm)'] >= SepWid_cuts[0]) & (iris_df_binary['sepal width (cm)'] < SepWid_cuts[1])
iris_df_binary['SepWid_2'] = iris_df_binary['SepWid_2'].astype(int)
iris_df_binary['SepWid_3'] = (iris_df_binary['sepal width (cm)'] >= SepWid_cuts[1]) & (iris_df_binary['sepal width (cm)'] < SepWid_cuts[2])
iris_df_binary['SepWid_3'] = iris_df_binary['SepWid_3'].astype(int)
iris_df_binary['SepWid_4'] = (iris_df_binary['sepal width (cm)'] >= SepWid_cuts[2])
iris_df_binary['SepWid_4'] = iris_df_binary['SepWid_4'].astype(int)
# drop the original column
iris_df_binary.drop(columns=['sepal width (cm)'], inplace=True)

# PetLen_cuts = [1.4, 4.35, 5.1]
PetLen_cuts = [1.4, 4.35, 5.1]
iris_df_binary['PetLen_1'] = (iris_df_binary['petal length (cm)'] < PetLen_cuts[0]) # to int
iris_df_binary['PetLen_1'] = iris_df_binary['PetLen_1'].astype(int)
iris_df_binary['PetLen_2'] = (iris_df_binary['petal length (cm)'] >= PetLen_cuts[0]) & (iris_df_binary['petal length (cm)'] < PetLen_cuts[1])
iris_df_binary['PetLen_2'] = iris_df_binary['PetLen_2'].astype(int)
iris_df_binary['PetLen_3'] = (iris_df_binary['petal length (cm)'] >= PetLen_cuts[1]) & (iris_df_binary['petal length (cm)'] < PetLen_cuts[2])
iris_df_binary['PetLen_3'] = iris_df_binary['PetLen_3'].astype(int)
iris_df_binary['PetLen_4'] = (iris_df_binary['petal length (cm)'] >= PetLen_cuts[2])
iris_df_binary['PetLen_4'] = iris_df_binary['PetLen_4'].astype(int)
# drop the original column
iris_df_binary.drop(columns=['petal length (cm)'], inplace=True)

# PetWid_cuts = [0.2, 1.3, 1.8]
PetWid_cuts = [0.2, 1.3, 1.8]
iris_df_binary['PetWid_1'] = (iris_df_binary['petal width (cm)'] < PetWid_cuts[0]) # to int
iris_df_binary['PetWid_1'] = iris_df_binary['PetWid_1'].astype(int)
iris_df_binary['PetWid_2'] = (iris_df_binary['petal width (cm)'] >= PetWid_cuts[0]) & (iris_df_binary['petal width (cm)'] < PetWid_cuts[1])
iris_df_binary['PetWid_2'] = iris_df_binary['PetWid_2'].astype(int)
iris_df_binary['PetWid_3'] = (iris_df_binary['petal width (cm)'] >= PetWid_cuts[1]) & (iris_df_binary['petal width (cm)'] < PetWid_cuts[2])
iris_df_binary['PetWid_3'] = iris_df_binary['PetWid_3'].astype(int)
iris_df_binary['PetWid_4'] = (iris_df_binary['petal width (cm)'] >= PetWid_cuts[2])
iris_df_binary['PetWid_4'] = iris_df_binary['PetWid_4'].astype(int)
# drop the original column
iris_df_binary.drop(columns=['petal width (cm)'], inplace=True)

# move target column to the end: target_0  target_1  target_2
iris_df_binary = iris_df_binary[[col for col in iris_df_binary.columns if col != 'target_0'] + ['target_0']]
iris_df_binary = iris_df_binary[[col for col in iris_df_binary.columns if col != 'target_1'] + ['target_1']]
iris_df_binary = iris_df_binary[[col for col in iris_df_binary.columns if col != 'target_2'] + ['target_2']]

# save to csv
iris_df_binary.to_csv('datasets/iris_binary.csv', index=False)

print(iris_df_binary.head())


# train a decisiont tree classifier on the binary dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder


# Load the dataset
iris_df_binary = pd.read_csv('datasets/iris_binary.csv')
print(iris_df_binary.head())

# Separate features and target
X = iris_df_binary.iloc[:, :-3].values  # All columns except the last three (target columns)
y = iris_df_binary.iloc[:, -3:].values  # Last three columns (target columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

"""
Accuracy: 0.93
Classification Report:
              precision    recall  f1-score   support

           0       0.83      1.00      0.91        10
           1       1.00      0.89      0.94         9
           2       1.00      0.91      0.95        11

    accuracy                           0.93        30
   macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30
"""

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
"""
Confusion Matrix:
[[10  0  0]
 [ 1  8  0]
 [ 1  0 10]]
"""
