# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:56:20 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.svm import SVR
import numpy as np
data = sio.loadmat('088IRWaSS7_Wi1d89_C4d3_wave')
datensatzen = 2000
datenlaenge =500
predictlen = 200
plt.close
testdatenstart =np.random.randint(10000,80000)
print(testdatenstart)
wave = data['WG10_DHI'][0][0][0]
wave = [i * 10 for i in wave]
length = np.arange(len(wave))
#plt.figure(1)
inputdata= []
#plt.plot(length,y1)
for i in range(datensatzen):
    for l in range (datenlaenge):
        
        inputdata.append(float(wave[i+l]))
inputdata = np.array(inputdata).reshape(datensatzen,datenlaenge)
inputlabel = np.zeros(shape =(datensatzen,predictlen))
for i in range(datensatzen):    
    for j in range(predictlen):
        inputlabel[i,j] = wave[i+datenlaenge+j]
xtest = []
for a in range(datenlaenge):
    xtest.append(float(wave[testdatenstart+a]))
xtest = np.array(xtest).reshape(1,-1)     
ypredict = []
ytest = wave[testdatenstart+datenlaenge:testdatenstart+datenlaenge+predictlen]
ytest = [ytest[i]for i in range(0,predictlen,4)]
for k in range (0,predictlen,4):
    clf = SVR(kernel = 'rbf', gamma=0.001, C=1000)
    inputy = inputlabel[:,k]
    clf.fit(inputdata,inputy)
    ypredict.append(clf.predict(xtest))

plt.plot(range(len(ytest)),ypredict,color = 'r', label = 'predict')
plt.plot(range(len(ytest)),ytest,color = 'b', label = 'real')