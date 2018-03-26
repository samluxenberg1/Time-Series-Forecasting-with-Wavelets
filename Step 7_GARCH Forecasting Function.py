# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:04:02 2017

@author: Samuel
"""

"""GARCH Forecasting Function"""

"""
Properties of GARCH Models (written in LATEX)
r_t = \mu_t + \epsilon_t
\epsilon_t = \sigma_t * e_t
\sigma_t^2 = \omega + \sum_{i=1}^{q}\alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{p}\beta \sigma_{t-j}^2 
where e_t ~ N(0,1) i.i.d., \epsilon_t ~ N(0, \sigma_t^2)
"""

def garch_forecast(data, index, split_date, p=1, q=1, forecast_length=1, horizon=1):
    
    import sys
    import numpy as np
    import pandas as pd
    import random
    from arch.univariate import arch_model
    
     
    #Model
    garch_mod = arch_model(data, vol='Garch', p=p, q=q)
    
    """Taken from examples in documentation"""
    
    #Recursive Forecast Generation
    data_df = pd.DataFrame(data, index=index)
    start_loc = 0
    end_loc = np.where(data_df.index >= split_date)[0].min()
    
    forecasts_mean = {}
    forecasts_variance = {}
    forecasts_residual_variance = {}
    
    for i in range(forecast_length):
        
        sys.stdout.write('.')
        sys.stdout.flush()
        
        #Fit the model
        res = garch_mod.fit(last_obs=i+end_loc-1, disp='off')
        
        #Create one-step forecasts for conditional mean, conditional variance, and conditional residual variance
        temp_mean = res.forecast(horizon=horizon).mean
        temp_variance = res.forecast(horizon=horizon).variance
        temp_residual_variance = res.forecast(horizon=horizon).residual_variance
                
        #Provide index of first most recent forecast
        fcast_mean = temp_mean.iloc[i+end_loc]
        fcast_variance = temp_variance.iloc[i+end_loc]                                             
        fcast_residual_variance = temp_residual_variance.iloc[i+end_loc]                        
        
        #Fill in forecasts dictionaries initialized before for loop
        forecasts_mean[fcast_mean.name] = fcast_mean
        forecasts_variance[fcast_variance.name] = fcast_variance
        forecasts_residual_variance[fcast_residual_variance.name] = fcast_residual_variance                          

    #Convert forecast dictionariies to data frames
    forecasts_mean_df = pd.DataFrame(forecasts_mean).T
    forecasts_variance_df = pd.DataFrame(forecasts_variance).T
    forecasts_residual_variance_df = pd.DataFrame(forecasts_residual_variance).T                                        

    #Forecast epsilon_t ~ N(0,sigma^2)
    forecasts_epsilon = np.zeros((len(forecasts_residual_variance_df),))   
    for j in range(forecast_length):
        forecasts_epsilon[j:j+1] = np.random.normal(0, forecasts_residual_variance_df[j:j+1], 1)
                                              
    #GARCH forecasts
    preds = np.ravel(forecasts_mean_df) + forecasts_epsilon                                            

    return preds                    