import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def genDataSet(N):
    x = np.random.normal(0, 1, N)
    ytrue = (np.cos(x) + 2) / (np.cos(x*1.4) +2)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    return x, y, ytrue
N = 100
x, y, ytrue = genDataSet(N)
x = np.array(x)
y =  np.array(y)

'''
for train_index,test_index in kf.split(x):
    #print("Train: ",train_index,"Test: ",test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
'''
i=1
k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
scores = []
cv_scores = []
#print("Train Index: ",len(train_index))
#print("Test Index: ",len(test_index))
for i in k:
    #print("K IS: ",i)
    neigh = KNeighborsRegressor(n_neighbors=i)
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    ytrue = ytrue.reshape((-1,1))
    neigh.fit(x, ytrue)
    score = neigh.score(x, ytrue)
    cv_score = cross_val_score(neigh, x, ytrue, cv=10)
    scores.append(score)
    cv_scores.append(np.mean(cv_score))
	
print ("R^2 scores: %s" %scores)
print ("CV scores: %s" %cv_scores)
plt.plot(x,y,'.')
plt.plot(x,ytrue,'rx')
#plt.show()
#print x
#print y
#print ytrue





