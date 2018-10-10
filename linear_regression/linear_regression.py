import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# fetch the dataset into pandas dataframe
df = pd.read_csv("linear.csv")

# initialize regressor
LR = LinearRegression()

# change features into right format
x = df.feature1.values.reshape(-1,1)
y = df.feature2.values.reshape(-1,1)

# train algorithm
LR.fit(x,y)

# predict the value of x = 2
print(LR.predict([[2]]))

# check the coefficient (ß1)
print(LR.coef_)

# check the intercept (ß0)
print(LR.intercept_)

# predict all the values in feature1
y_ = LR.predict(x)

# plot them on screen
plt.scatter(df.feature1, df.feature2, color = 'green')
plt.plot(x, y_, color = 'red')
plt.show()
