
import numpy as np
import pandas as pd

from sklearn import datasets

digits = datasets.load_digits()

X = digits.data
y = digits.target



print(X.shape, y.shape)
print("X", X[0])
print("Y", y[0])

threshold = 7
X = np.where(X > threshold, 1, 0)
print(X[0])

# 64 features and 1 target
# one hot encoding

y_onehot = np.zeros((y.shape[0], 10))
y_onehot[np.arange(y.shape[0]), y] = 1

print("Y one hot", y_onehot[0])

print("X", X[0])
print("Y", y[0])

# write the binary features and one hot encoded target to a csv file
df = pd.DataFrame(X)

# One-hot encode the target
for i in range(10):
    df[f'digit_{i}'] = (y == i).astype(int)

df.to_csv('digits.csv', index=False)

# filter on rows that only have 0 and 1
