from sklearn.cluster import KMeans
import numpy as np

dataname = r'ZIFA/Z-biase.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
clf = KMeans(n_clusters=4)
sum=0
for i in range(50):
    s = clf.fit(x)
    C = s.labels_

np.savetxt('ZIFA/biase.txt', C, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/camp.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/camp.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/deng.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/deng.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/goolam.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/goolam.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/kolo.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/kolo.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/li.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/li.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/pollen.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/pollen.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/usoskin.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/usoskin.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')

import numpy as np

dataname = r'SIMLR/yan.csv'
with open(dataname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
np.savetxt('SIMLR/yan.txt', y, encoding='utf-8', delimiter='/n', fmt='%d')
