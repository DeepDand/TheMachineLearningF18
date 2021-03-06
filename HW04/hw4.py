import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split

def genDataSet(N):
    x = np.random.normal(0, 1, N)
    ytrue = (np.cos(x) + 2) / (np.cos(x*1.4) +2)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    return x, y, ytrue
N = 1000
x, y, ytrue = genDataSet(N)
x = np.array(x)
y =  np.array(y)
x = x.reshape((1,-1))
y = y.reshape((1,-1))
knn.fit(X_train, y_train)
plt.plot(x,y,'.')
plt.plot(x,ytrue,'rx')
#plt.show()
#print x
#print y
#print ytrue
