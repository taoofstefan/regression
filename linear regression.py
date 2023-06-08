# Linear regression with Python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import datasets
from sklearn import linear_model

# Load dataset
bostonData = datasets.load_boston()

yb = bostonData.target.reshape(-1, 1)
Xb = bostonData ['data'][:,5].reshape(-1, 1)

plt.scatter(Xb,yb)
plt.ylabel ('value of house /1000')
plt.xlabel ('number of')
plt.show

regr = linear_model.LinearRegression () # Create the model
regr.fit(Xb, yb) # Train the model

plt.scatter(Xb, yb, color= 'black')
plt.plot(Xb, regr.predict(Xb), color='blue', linewidth=3)
plt.show