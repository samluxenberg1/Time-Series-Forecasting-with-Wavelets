# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:03:14 2017

@author: Samuel
"""

"""Gas Prices Data Processing"""
import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  
from datetime import datetime
import math
import scipy as sp
import sklearn
from sklearn.neural_network import MLPRegressor

GasPrices = pd.read_csv("C:\\Users\\Samuel\\PycharmProjects\\iknowsecond\\GasPrices.csv")
GasPrices.columns = ['Week', 'Price']    

#Want data from January 2000 to November 2013

#Find the indices of each
GasPrices['Week'][GasPrices['Week'] == "1/7/2000"].index
GasPrices['Week'][GasPrices['Week'] == "11/29/2013"].index         

data = GasPrices[155:880]         
data.set_index(['Week'], inplace = True)
#data_ts = data.reindex(index = data.index[::-1])
data.index = pd.to_datetime(data.index, format = '%m/%d/%Y')

plt.plot(data)
###################################plt.plot(test_ts)
plt.title('Weekly Henry Hub Natural Gas Spot Prices', fontsize=30)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Dollars per Million Btu', fontsize=20)
         
#Use first 624 observations to build models
data_ts = data[0:624]
#Use the last 101 observations to test forecasting performance
test_ts = data[624:]

#Reshape
data_ts_np = data_ts.as_matrix()
data_ts1 = data_ts_np.reshape((len(data_ts_np,)))

test_ts_np = test_ts.as_matrix()
test_ts1 = test_ts_np.reshape((len(test_ts_np,)))

