# Logistic regression with Python
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import datasets , linear_model
import numpy as np
X1 = np.random.normal(size=150)
y1 = (X1 > 0).astype(np.float64)
X1[X1 > 0] *= 4
X1 += .3 * np.random.normal(size=150)
X1= X1.reshape(-1, 1)

plt.scatter(X1,y1)
plt.ylabel('y1')
plt.xlabel('X1')
plt.show

lm_log = linear_model.LogisticRegression()
lm_log.fit (X1, y1)
X1_ordered = np.sort(X1, axis=0)

plt.scatter(X1.ravel(), y1, color='black', zorder=20 , alpha=0.5)
plt.plot(X1_ordered, lm_log.predict_proba(X1_ordered)[:,1], color='blue', linewidth = 3)
plt.ylabel('target variable')
plt.xlabel('predictor variable')
plt.show()