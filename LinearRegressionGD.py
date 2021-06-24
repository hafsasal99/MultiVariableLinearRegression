import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data read and split into features and labels
df = pd.read_csv('ex1data2.txt', sep=",", header=None)
x, y = df.iloc[:, :-1], df.iloc[:, [-1]]

# normalizing data
means = pd.DataFrame(x).mean()
sigmas = pd.DataFrame(x).std()
x = (x-means)/sigmas

# appending a dummy column to features matrix
x.insert(0,'dummy', 1)

# weights
weights=np.random.rand((len(x.columns)))

# linear regression
sample_Size = len(x.index)
x = x.transpose()
predicted = np.dot(weights.transpose(), x)
predicted = pd.Series(predicted)

# cost function
y = y.squeeze()  # converting the labels into a pandas series
errorSquared = np.square(y.subtract(predicted))
cost = 1/2 * sample_Size * sum(errorSquared)

# gradient descent
count = 0
trainingRate = 0.01
costlist = []
while cost != 0 and count < 1000 :
    # weight update rule
    weights = pd.Series(weights)
    featureVector = x.transpose()
    new_weights = weights.subtract(trainingRate * sum(np.dot((predicted.subtract(y)).to_frame().transpose(),featureVector)).transpose() * 1 / len(x.index))
    weights = new_weights
    print(weights)
    predicted = np.dot(weights.transpose(), x)
    predicted = pd.Series(predicted)
    errorSquared = np.square(y.subtract(predicted))
    cost = 1 / 2 * sample_Size * sum(errorSquared)
    if count % 5 == 0:
        costlist.append(cost)
    count += 1
    print(cost)

# Convergence Curve
costlist = np.asarray(costlist)
count=list(range(0, count, 5))
count=np.asarray(count)
plt.plot(count, costlist, '-r')  # plot the cost function.
plt.show()

# testing data
test_features=[1,1650.0,3.0]
test_features[1] = (test_features[1]-means[0])/sigmas[0]
test_features[2] = (test_features[2]-means[1])/sigmas[1]
test_features=np.array(test_features)
test_features = test_features.transpose()
predicted = np.dot(weights.transpose(), test_features)
print('The predicted price of this house is ', predicted)








