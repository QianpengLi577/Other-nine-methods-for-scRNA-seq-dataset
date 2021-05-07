import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from ACC import ACC

labelname='yan'
y = np.loadtxt('label/'+labelname+'.csv', encoding='utf-8', delimiter=',').astype(np.int)
y_corr = np.loadtxt('CORR/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)
y_cos = np.loadtxt('COS/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)
y_eu = np.loadtxt('EU/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)
y_pocr = np.loadtxt('POCR/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)
y_simlr = np.loadtxt('SIMLR/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)
y_sinnlrr= np.loadtxt('SinNLRR/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)
y_sp = np.loadtxt('SP/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)
y_snncliq = np.loadtxt('SNNCliq/'+labelname+'.txt', encoding='utf-8', delimiter='/n').astype(np.int)

ACCbiase = []
ACCbiase.append(ACC(y,y_corr))
ACCbiase.append(ACC(y,y_cos))
ACCbiase.append(ACC(y,y_eu))
ACCbiase.append(ACC(y,y_pocr))
ACCbiase.append(ACC(y,y_simlr))
ACCbiase.append(ACC(y,y_sinnlrr))
ACCbiase.append(ACC(y,y_sp))
ACCbiase.append(ACC(y,y_snncliq))

NMIbiase = []
NMIbiase.append(NMI(y,y_corr))
NMIbiase.append(NMI(y,y_cos))
NMIbiase.append(NMI(y,y_eu))
NMIbiase.append(NMI(y,y_pocr))
NMIbiase.append(NMI(y,y_simlr))
NMIbiase.append(NMI(y,y_sinnlrr))
NMIbiase.append(NMI(y,y_sp))
NMIbiase.append(NMI(y,y_snncliq))

ARIbiase = []
ARIbiase.append(ARI(y,y_corr))
ARIbiase.append(ARI(y,y_cos))
ARIbiase.append(ARI(y,y_eu))
ARIbiase.append(ARI(y,y_pocr))
ARIbiase.append(ARI(y,y_simlr))
ARIbiase.append(ARI(y,y_sinnlrr))
ARIbiase.append(ARI(y,y_sp))
ARIbiase.append(ARI(y,y_snncliq))

biase = np.vstack((np.array(ARIbiase),np.array(NMIbiase)))
biase = np.vstack((biase, np.array(ACCbiase)))
np.savetxt(labelname+'.csv',biase,encoding='utf-8', delimiter=',', fmt='%f')
print('finsh')