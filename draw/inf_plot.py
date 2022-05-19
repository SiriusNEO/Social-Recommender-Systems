import os
from statistics import mode
import sys
sys.path.append(os.path.abspath('../src/'))

from dataloader import DataLoader
from TriSR import TriSR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import random as rd

loader = DataLoader()

loader.read_epinions(split=False)

model = TriSR(
            train_set = loader.trainSet,
            test_set = loader.testSet,
            rating = loader.ratingList,
            rset = loader.ratingSet,
            network = loader.adjList,
            m = loader.userNum, 
            n = loader.itemNum, 
            l = 10,
            alpha=2, beta=5, gamma=3)

model.user_rank(10, 0.85)

plt.xlabel("UserID")
plt.ylabel("Influence")
plt.plot(range(1, len(model.rank)+1), model.rank)

plt.show()

"""
star = 0

for i in range(model.m):
    if len(model.network[i]) > len(model.network[star]):
        star = i

X = []
Y = []

for i in range(model.m):
    sim = model.PCC(i, star)
    dsim = np.exp(-sim)-np.exp(-1)
    vec = np.array([model.alpha*model.rank[i] + model.beta*sim + model.gamma*model.fam(i, star)])
    dfam = np.exp(-model.fam(star, i))-np.exp(-1)
    deg = rd.randint(1, 360)
    dvec = np.exp(-np.linalg.norm(vec))-np.exp(-1) if i != star else 0
    X.append(dvec*np.cos(deg/360*2*np.pi))
    Y.append(dvec*np.sin(deg/360*2*np.pi))

# plt.scatter(X, Y)
# plt.plot(range(model.m), model.rank)


plt.hist2d(X, Y, bins = 50,  
           cmap = "binary", 
           norm = colors.LogNorm())

plt.axis('off')
plt.colorbar(shrink=.83)

plt.show()
"""