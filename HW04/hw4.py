import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

def genDataSet(N):
    x = np.random.normal(0, 1, N)
    ytrue = (np.cos(x) + 2) / (np.cos(x*1.4) +2)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    return x, y, ytrue

x, y, ytrue = genDataSet(1000)
x = np.array(x)
y =  np.array(y)
kf = KFold(n_splits=10)
kf.get_n_splits(x)
print("kf: ",kf)

#k_fold = KFold(n_splits=10, random_state=None, shuffle=False)

for train_index,test_index in kf.split(x):
    print("Train: ",train_index,"Test: ",test_index)
plt.plot(x,y,'.')
plt.plot(x,ytrue,'rx')
#plt.show()
#print x
#print y
#print ytrue
x = x.reshape((1,-1))
y = y.reshape((1,-1))


neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(x,y)

