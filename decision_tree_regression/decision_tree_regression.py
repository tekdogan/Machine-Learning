import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# fetch the dataset into pandas dataframe
df = pd.read_csv('cart_dataset.csv')

# change the features into right format
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# initialize the regressor
tree_regressor = DecisionTreeRegressor()

# train the algorithm
tree_regressor.fit(x,y)

# recreate the feature1 with frequency of 0.01
x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)

# and predict all values of them
y_ = tree_regressor.predict(x_)

# plot them all on the same graph
plt.scatter(x, y, color = 'green')
plt.plot(x_, y_, color = 'red')
plt.title("Decision Tree")
plt.show()
