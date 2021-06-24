import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# data read and split into features and labels
df = pd.read_csv('ex1data2.txt', sep=",", header=None)
x, y = df.iloc[:, :-1], df.iloc[:, [-1]]


# appending a dummy column to features matrix
x.insert(0,'dummy', 1)

# normal equation
x_transpose=np.transpose(x)
equation = np.linalg.pinv(np.dot(x_transpose,x)) @ x_transpose @ y
print(equation)

#prediction
input=np.array([1, 1650, 3])
predicted = input.transpose() @ equation
print('The predicted price of this house is ', predicted.to_string(index=False))


