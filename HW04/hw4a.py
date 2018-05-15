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
N = 1000
x, y, ytrue = genDataSet(N)
plt.plot(x,y,'.')
plt.plot(x,ytrue,'rx')
plt.savefig('HW4_PartA', dpi=200, bbox_inches='tight')





