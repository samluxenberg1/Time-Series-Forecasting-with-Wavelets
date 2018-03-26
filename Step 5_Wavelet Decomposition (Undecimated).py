# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:17:38 2017

@author: Samuel
"""

"""Wavelet Decomposition"""

#Decompose training data using undecimated wavelet transform
(cA3_train,cD3_train), (cA2_train,cD2_train), (cA1_train,cD1_train) = pywt.swt(np.ravel(data_ts1), 'db3', level = 3)

#Decompose all data using undecimated wavelet transform (need even number of points)
data_pad = np.lib.pad(np.ravel(data), (0, 299), 'constant', constant_values=0)
(cA3,cD3), (cA2,cD2), (cA1,cD1) = pywt.swt(data_pad, 'db3', level = 3)

"""
Adjusted A-Trous Algorithm

Given a series S(0), S(1), ..., S(99)

1. Carry out discrete wavelet decomposition on S(0), S(1), ..., S(k)
2. Retain the decomposed values for the (k-1)st and kth time points only
3. If k < 100, set k = k + 2 and repeat these steps


retain_dec=[]
k=0
while (k < len(data_ts1)):
    temp_cA, temp_cD = pywt.dwt(data_ts1[:k+2], 'db3')
    retain_dec = retain_dec + [temp_dec[-2:]]
    k = k + 2
    

Getting an error that the decomposition level=3 is too high. for k=0, only get 2 elements
so the max decomposition level would be 1. How did the authors carry this out?
  
"""