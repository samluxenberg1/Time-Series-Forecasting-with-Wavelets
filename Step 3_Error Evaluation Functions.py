# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:38:16 2017

@author: Samuel
"""

"""Error Evaluation Functions"""
def MSE(forecasts, original):
    import math
    
    #Mean-Squared Error
    error = forecasts - original
    error_sq = error**2
    mse = error_sq.mean()
    
    #Root-Mean-Squared Error
    rmse = math.sqrt(mse)
    
    print("MSE is: ", mse)
    print("RMSE is: ", rmse)
    
    return mse, rmse

def MAPE(forecasts, original):
    
    #Mean Absolute Percentage Error
    ape = abs((forecasts - original)/original)
    mape = ape.mean()
    
    print("MAPE is: ", mape)
    
    return mape

