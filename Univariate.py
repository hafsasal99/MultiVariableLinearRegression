import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data read and split into features and labels
df = pd.read_csv('ex1data1.txt', sep=",", header=None)
x, y = df.iloc[:, :-1], df.iloc[:, [-1]]

# appending a dummy column to features matrix
x.insert(0,'dummy', 1)

# weights
weights=np.random.rand((len(x.columns)))

# linear regression
predicted = np.dot(x,weights.transpose())
predicted = pd.Series(predicted)

# cost function
y = y.squeeze()  # converting the labels into a pandas series
errorSquared = np.square(y.subtract(predicted))
cost = 1/2 * len(x.index) * sum(errorSquared)

# gradient descent
count = 0
trainingRate = 0.005
costlist = []
while cost != 0 and count < 50 :
    # weight update rule
    weights = pd.Series(weights)
    new_weights = weights.subtract(trainingRate * sum(np.dot((predicted.subtract(y)).to_frame().transpose(),x)) * 1 / len(x.index))
    weights = new_weights
    print(weights)
    predicted = np.dot(x, weights.transpose())
    predicted = pd.Series(predicted)
    errorSquared = np.square(y.subtract(predicted))
    cost = 1 / 2 * len(x.index) * sum(errorSquared)
    if count % 5 == 0:
        costlist.append(cost)
    count += 1
    print(cost)
costlist = np.asarray(costlist)
count=list(range(0, count, 5))
count=np.asarray(count)
plt.plot(count, costlist, '-r')  # plot the cost function.

plt.show()






