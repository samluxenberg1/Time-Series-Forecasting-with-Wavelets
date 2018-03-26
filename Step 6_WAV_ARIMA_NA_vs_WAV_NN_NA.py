# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:01:52 2017

@author: Samuel
"""

"""Wav-ARIMA-N/A vs. Wav-ANN-N/A"""

###Wav-ARIMA-N/A

#Forecast cA3 with ARIMA (3,1,1)
arima_cA3_train_mod = sm.tsa.SARIMAX(cA3_train, order=(3,1,1))
arima_cA3_train_res = arima_cA3_train_mod.fit(maxiter=100)

arima_cA3 = sm.tsa.SARIMAX(cA3, order=(3,1,1))
arima_cA3_res = arima_cA3.filter(arima_cA3_train_res.params)

arima_cA3_preds = arima_cA3_res.predict(start=624, end=724)

#Reconstruct from forecasted coefficients
cA3_for_recon = cA3.copy()
cA3_for_recon[624:725] = arima_cA3_preds
cA3_recon = pywt.iswt([(cA3_for_recon, np.zeros((len(cD3),))), (np.zeros((len(cA2),)), np.zeros((len(cD2),))), (np.zeros((len(cA1),)), np.zeros((len(cD1),)))], 'db3')             

#Evaluate
MSE(cA3_recon[627:725], data_pad[627:725])
MAPE(cA3_recon[627:725], data_pad[627:725])


###Wav-ANN-N/A

#Forecast cA3 with ANN: Time-lags = 3, Nodes = 4
cA3_preds, cA3_preds_unstd, cA3_MSE_std, cA3_RMSE_std, cA3_MSE, cA3_RMSE = NeuralNet_ts_predictions(cA3_train, cA3[624:725], num_lags=3, hidden_layer_sizes=(4, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=42, tol=1e-10, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-10)

#Reconstruct from forecasted coefficients
cA3_for_recon_nn = cA3.copy()
cA3_for_recon_nn[627:725] = cA3_preds_unstd
cA3_recon_nn = pywt.iswt([(cA3_for_recon_nn, np.zeros((len(cA3_for_recon_nn),))), (np.zeros((len(cA2),)), np.zeros((len(cD2),))), (np.zeros((len(cA1),)), np.zeros((len(cD1),)))], 'db3')

#Evaluate
MSE(cA3_recon_nn[627:725], data_pad[627:725])
MAPE(cA3_recon_nn[627:725], data_pad[627:725])

"""Comparison Plot"""
cA3_recon_df = pd.DataFrame(cA3_recon[627:725], index=data[627:725].index)
cA3_recon_nn_df = pd.DataFrame(cA3_recon_nn[627:725], index=data[627:725].index)

plt.plot(data[627:725], label='Actual')
plt.plot(cA3_recon_df, label='WAV-ARIMA')
plt.plot(cA3_recon_nn_df, label='WAV-ANN')
plt.legend(loc='upper left', prop={'size':24})
plt.title('Approximation Forecasts', fontsize=30)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Dollars per Million Btu', fontsize=20)