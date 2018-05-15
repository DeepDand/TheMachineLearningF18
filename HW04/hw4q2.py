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
N = 1000
x, y, ytrue = genDataSet(N)
x = np.array(x)
y =  np.array(y)
kf = KFold(n_splits=10)
kf.get_n_splits(x)

for train_index,test_index in kf.split(x):
    #print("Train: ",train_index,"Test: ",test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

i=1
k_val = []
scores = []
cv_scores = []
#print("Train Index: ",len(train_index))
#print("Test Index: ",len(test_index))
k_each = []
k_all = []

for j in xrange(1,100,1):
    for i in xrange(1, len(train_index),2):
        #print("K IS: ",i)
        k_val.append(i) 
        neigh = KNeighborsRegressor(n_neighbors=i)
        x = x.reshape((-1,1))
        y = y.reshape((-1,1))
        ytrue = ytrue.reshape((-1,1))
        neigh.fit(x, ytrue)
        score = neigh.score(x, ytrue)
        cv_score = cross_val_score(neigh, x, ytrue, cv=10)
        scores.append(score)
        cv_scores.append(np.mean(cv_score))
        #print cv_score
	
#print ("R^2 scores: %s" %scores)
#print ("CV scores: %s" %cv_scores)
    d = {'k_value': k_val,'CV Score': cv_scores}
    results = pd.DataFrame(data=d)
    max_e = max(results["CV Score"])
    #print "The best results are: %f \n ",results.loc[(results["CV Score"]== max_e)]

    print "The three best results are: %f \n ",results.loc[results["CV Score"].argsort()[-3:]]
    print "J is ", j
    #print results.loc[results["CV Score"].argsort()[-3:],'k_value']	
    k_each = results.loc[results["CV Score"].argsort()[-3:],'k_value'].iloc[0]
    k_each = k_each.append(results.loc[results["CV Score"].argsort()[-2:],'k_value'].iloc[0])
	#k_each = results.loc[results["CV Score"].argsort()[-3:],'k_value'].iloc[0]
    k_all.append(k_each)
    results = results.reset_index(drop=True, inplace=True)
    #del results['index']
    #results.drop(results.index, inplace=True)

print k_all




