import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from ACC import ACC
from sklearn.cluster import KMeans

# label
y_biase = np.loadtxt('label/biase.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_camp = np.loadtxt('label/camp.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_deng = np.loadtxt('label/deng.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_goolam = np.loadtxt('label/goolam.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_kolo = np.loadtxt('label/kolo.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_li = np.loadtxt('label/li.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_pollen = np.loadtxt('label/pollen.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_usoskin = np.loadtxt('label/usoskin.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_yan = np.loadtxt('label/yan.csv', encoding='utf-8', delimiter=',').astype(np.int)
# classes
n_biase = y_biase.max() - y_biase.min() + 1
n_camp = y_camp.max() - y_camp.min() + 1
n_deng = y_deng.max() - y_deng.min() + 1
n_goolam = y_goolam.max() - y_goolam.min() + 1
n_kolo = y_kolo.max() - y_kolo.min() + 1
n_li = y_li.max() - y_li.min() + 1
n_pollen = y_pollen.max() - y_pollen.min() + 1
n_usoskin = y_usoskin.max() - y_usoskin.min() + 1
n_yan = y_yan.max() - y_yan.min() + 1
# ZIFA data
x_biase = np.loadtxt('ZIFA/Z-biase.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_camp = np.loadtxt('ZIFA/Z-camp.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_deng = np.loadtxt('ZIFA/Z-deng.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_goolam = np.loadtxt('ZIFA/Z-goolam.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_kolo = np.loadtxt('ZIFA/Z-kolo.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_li = np.loadtxt('ZIFA/Z-li.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_pollen = np.loadtxt('ZIFA/Z-pollen.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_usoskin = np.loadtxt('ZIFA/Z-usoskin.csv', encoding='utf-8', delimiter=',').astype(np.float)
x_yan = np.loadtxt('ZIFA/Z-yan.csv', encoding='utf-8', delimiter=',').astype(np.float)

# biase part
clf_biase = KMeans(n_clusters=n_biase)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_biase)
    C = s.labels_
    ARI_biase += ARI(y_biase, C)
    NMI_biase += NMI(y_biase, C)
    ACC_biase += ACC(y_biase, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_biase.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('biasefinsh')


# camp part
clf_biase = KMeans(n_clusters=n_camp)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_camp)
    C = s.labels_
    ARI_biase += ARI(y_camp, C)
    NMI_biase += NMI(y_camp, C)
    ACC_biase += ACC(y_camp, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_camp.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('campfinsh')

# deng part
clf_biase = KMeans(n_clusters=n_deng)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_deng)
    C = s.labels_
    ARI_biase += ARI(y_deng, C)
    NMI_biase += NMI(y_deng, C)
    ACC_biase += ACC(y_deng, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_deng.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('dengfinsh')

# goolam part
clf_biase = KMeans(n_clusters=n_goolam)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_goolam)
    C = s.labels_
    ARI_biase += ARI(y_goolam, C)
    NMI_biase += NMI(y_goolam, C)
    ACC_biase += ACC(y_goolam, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_goolam.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('goolamfinsh')

# kolo part
clf_biase = KMeans(n_clusters=n_kolo)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_kolo)
    C = s.labels_
    ARI_biase += ARI(y_kolo, C)
    NMI_biase += NMI(y_kolo, C)
    ACC_biase += ACC(y_kolo, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_kolo.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('kolofinsh')

# li part
clf_biase = KMeans(n_clusters=n_li)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_li)
    C = s.labels_
    ARI_biase += ARI(y_li, C)
    NMI_biase += NMI(y_li, C)
    ACC_biase += ACC(y_li, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_li.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('lifinsh')

# pollen part
clf_biase = KMeans(n_clusters=n_pollen)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_pollen)
    C = s.labels_
    ARI_biase += ARI(y_pollen, C)
    NMI_biase += NMI(y_pollen, C)
    ACC_biase += ACC(y_pollen, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_pollen.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('pollenfinsh')

# usoskin part
clf_biase = KMeans(n_clusters=n_usoskin)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_usoskin)
    C = s.labels_
    ARI_biase += ARI(y_usoskin, C)
    NMI_biase += NMI(y_usoskin, C)
    ACC_biase += ACC(y_usoskin, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_usoskin.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('usoskinfinsh')

# yan part
clf_biase = KMeans(n_clusters=n_yan)
ARI_biase = 0
NMI_biase = 0
ACC_biase = 0
for i in range(50):
    s = clf_biase.fit(x_yan)
    C = s.labels_
    ARI_biase += ARI(y_yan, C)
    NMI_biase += NMI(y_yan, C)
    ACC_biase += ACC(y_yan, C)
ARI_biase =ARI_biase*1.0/50
NMI_biase =NMI_biase*1.0/50
ACC_biase =ACC_biase*1.0/50
biase = np.vstack((np.array(ARI_biase), np.array(NMI_biase)))
biase = np.vstack((biase, np.array(ACC_biase)))
np.savetxt('ZIFA_yan.csv', biase, encoding='utf-8', delimiter=',', fmt='%f')
print('yanfinsh')