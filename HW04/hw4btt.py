import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
def genDataSet(N):
    x = np.random.normal(0, 1, N)
    ytrue = (np.cos(x) + 2) / (np.cos(x*1.4) +2)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    return x, y, ytrue
N = 1000 #defining N 
x, y, ytrue = genDataSet(N)

#converting x, y, ytrue to NP Arrays
x = np.array(x) 
y =  np.array(y)

#using KFold to get the splits
kf = KFold(n_splits=10)
kf.get_n_splits(x)

#dividing the data in train and test sets
for train_index,test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

i=1
k_val = []
scores = []
cv_scores = []
for i in xrange(1, len(train_index),2):
    k_val.append(i) 
    #performing KNN Regressions
    neigh = KNeighborsRegressor(n_neighbors=i)
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    ytrue = ytrue.reshape((-1,1))
    neigh.fit(x, ytrue)
    #checking the R square (goodness of fit) for the knn
    score = neigh.score(x, ytrue)
    #checking cross validation score
    cv_score = cross_val_score(neigh, x, ytrue, cv=10)
    scores.append(score)
    cv_scores.append(np.mean(cv_score))
	
d = {'k_value': k_val,'CV Score': cv_scores}
results = pd.DataFrame(data=d)
print "The three best results are: %f \n ",results.loc[results["CV Score"].argsort()[-3:]]




