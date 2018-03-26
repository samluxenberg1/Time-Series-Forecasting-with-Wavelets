# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:00:50 2017

@author: Samuel
"""

"""
Detail Coefficient Models

GARCH Models
    -cD1: GARCH(3,3)
    -cD2: GARCH(1,2)
    -cD3: GARCH(1,2)

Models: 
    -Wav-ARIMA-GARCH
    -Wav-ARIMA-ANN
    -Wav-ANN-GARCH
    -Wav-ANN-ANN
"""
#GARCH forecasts of detail coefficients
np.random.seed(42)
cD1_preds = garch_forecast(cD1[:725], index=data[:725].index, split_date='2011-12-30', p=3, q=3, forecast_length=101, horizon=1)
np.random.seed(42)
cD2_preds = garch_forecast(cD2[:725], index=data[:725].index, split_date='2011-12-30', p=1, q=2, forecast_length=101, horizon=1)
np.random.seed(42)
cD3_preds = garch_forecast(cD3[:725], index=data[:725].index, split_date='2011-12-30', p=1, q=2, forecast_length=101, horizon=1)

#ANN forecasts of detail coefficients
cD1_nn_preds, cD1_nn_preds_unstd, cD1_MSE_std, cD1_RMSE_std, cD1_MSE, cD1_RMSE = NeuralNet_ts_predictions(cD1_train, cD1[624:725], num_lags=3, hidden_layer_sizes=(3,), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=42, tol=1e-10, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-10)
cD2_nn_preds, cD2_nn_preds_unstd, cD2_MSE_std, cD2_RMSE_std, cD2_MSE, cD2_RMSE = NeuralNet_ts_predictions(cD2_train, cD2[624:725], num_lags=2, hidden_layer_sizes=(4,), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=42, tol=1e-10, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-10)
cD3_nn_preds, cD3_nn_preds_unstd, cD3_MSE_std, cD3_RMSE_std, cD3_MSE, cD3_RMSE = NeuralNet_ts_predictions(cD3_train, cD3[624:725], num_lags=2, hidden_layer_sizes=(5,), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=42, tol=1e-10, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-10)


#Prepare coefficient forecasts for wavelet reconstruction
cD1_for_recon = cD1.copy()
cD2_for_recon = cD2.copy()
cD3_for_recon = cD3.copy()

cD1_for_recon_nn = cD1.copy()
cD2_for_recon_nn = cD2.copy()
cD3_for_recon_nn = cD3.copy()

cD1_for_recon[624:725] = cD1_preds
cD2_for_recon[624:725] = cD2_preds
cD3_for_recon[624:725] = cD3_preds             

cD1_for_recon_nn[627:725] = cD1_nn_preds_unstd
cD2_for_recon_nn[626:725] = cD2_nn_preds_unstd
cD3_for_recon_nn[626:725] = cD3_nn_preds_unstd                
             
###Wav-ARIMA-GARCH
wav_arima_garch_recon = pywt.iswt([(cA3_for_recon, cD3_for_recon), (np.zeros((len(cA2),)), cD2_for_recon), (np.zeros((len(cA1),)), cD1_for_recon)], 'db3')             

#Evaluate
MSE(wav_arima_garch_recon[627:725], data_pad[627:725])
MAPE(wav_arima_garch_recon[627:725], data_pad[627:725])

###Wav-ANN-GARCH
wav_ann_garch_recon = pywt.iswt([(cA3_for_recon_nn, cD3_for_recon), (np.zeros((len(cA2),)), cD2_for_recon), (np.zeros((len(cA1),)), cD1_for_recon)], 'db3')             

#Evaluate
MSE(wav_ann_garch_recon[627:725], data_pad[627:725])
MAPE(wav_ann_garch_recon[627:725], data_pad[627:725])

###Wav-ARIMA-ANN
wav_arima_ann_recon = pywt.iswt([(cA3_for_recon, cD3_for_recon_nn), (np.zeros((len(cA2),)), cD2_for_recon_nn), (np.zeros((len(cA1),)), cD1_for_recon_nn)], 'db3')             

#Evaluate
MSE(wav_arima_ann_recon[627:725], data_pad[627:725])
MAPE(wav_arima_ann_recon[627:725], data_pad[627:725])

###Wav-ANN-ANN(cD3,cD2,cD1)
wav_ann_ann_recon = pywt.iswt([(cA3_for_recon_nn, cD3_for_recon_nn), (np.zeros((len(cA2),)), cD2_for_recon_nn), (np.zeros((len(cA1),)), cD1_for_recon_nn)], 'db3')             

#Evaluate
MSE(wav_ann_ann_recon[627:725], data_pad[627:725])
MAPE(wav_ann_ann_recon[627:725], data_pad[627:725])

###Wav-ANN-ANN(cD3)
wav_ann_cD3nn = pywt.iswt([(cA3_for_recon_nn, cD3_for_recon_nn), (np.zeros((len(cA2),)), np.zeros((len(cA2),))), (np.zeros((len(cA1),)), np.zeros((len(cA1),)))], 'db3')

#Evaluate
MSE(wav_ann_cD3nn[627:725], data_pad[627:725])
MAPE(wav_ann_cD3nn[627:725], data_pad[627:725])

###Wav-ANN-ANN(cD3,cD2)
wav_ann_cD3cD2nn = pywt.iswt([(cA3_for_recon_nn, cD3_for_recon_nn), (np.zeros((len(cA2),)), cD2_for_recon_nn), (np.zeros((len(cA1),)), np.zeros((len(cA1),)))], 'db3')

#Evaluate
MSE(wav_ann_cD3cD2nn[627:725], data_pad[627:725])
MAPE(wav_ann_cD3cD2nn[627:725], data_pad[627:725])

###Wav-ANN-ANN(cD2)
wav_ann_cD2nn = pywt.iswt([(cA3_for_recon_nn, np.zeros((len(cA3),))), (np.zeros((len(cA2),)), cD2_for_recon_nn), (np.zeros((len(cA1),)), np.zeros((len(cA1),)))], 'db3')

#Evaluate
MSE(wav_ann_cD2nn[627:725], data_pad[627:725])
MAPE(wav_ann_cD2nn[627:725], data_pad[627:725])

###Wav-ANN-ANN(cD1)
wav_ann_cD1nn = pywt.iswt([(cA3_for_recon_nn, np.zeros((len(cA3),))), (np.zeros((len(cA2),)), np.zeros((len(cA2),))), (np.zeros((len(cA1),)), cD1_for_recon_nn)], 'db3')

#Evaluate
MSE(wav_ann_cD1nn[627:725], data_pad[627:725])
MAPE(wav_ann_cD1nn[627:725], data_pad[627:725])

###Wav-ANN-ANN(cD3,cD1)
wav_ann_cD3cD1nn = pywt.iswt([(cA3_for_recon_nn, cD3_for_recon_nn), (np.zeros((len(cA2),)), np.zeros((len(cA2),))), (np.zeros((len(cA1),)), cD1_for_recon_nn)], 'db3')

#Evaluate
MSE(wav_ann_cD3cD1nn[627:725], data_pad[627:725])
MAPE(wav_ann_cD3cD1nn[627:725], data_pad[627:725])

###Wav-ANN-ANN(cD2,cD1)
wav_ann_cD2cD1nn = pywt.iswt([(cA3_for_recon_nn, np.zeros((len(cD3),))), (np.zeros((len(cA2),)), cD2_for_recon_nn), (np.zeros((len(cA1),)), cD1_for_recon_nn)], 'db3')

#Evaluate
MSE(wav_ann_cD2cD1nn[627:725], data_pad[627:725])
MAPE(wav_ann_cD2cD1nn[627:725], data_pad[627:725])

"""Comparison Plots"""
wav_arima_garch_recon_df = pd.DataFrame(wav_arima_garch_recon[627:725], index=data[627:725].index)
wav_ann_garch_recon_df = pd.DataFrame(wav_ann_garch_recon[627:725], index=data[627:725].index)
wav_arima_ann_recon_df = pd.DataFrame(wav_arima_ann_recon[627:725], index=data[627:725].index)
wav_ann_ann_recon_df = pd.DataFrame(wav_ann_ann_recon[627:725], index=data[627:725].index)

plt.plot(data[627:725], color='blue', label='Actual')
plt.plot(wav_arima_garch_recon_df, color='red', label='WAV-ARIMA-GARCH')
plt.plot(wav_ann_garch_recon_df, color='green', label='WAV-ANN-GARCH')
plt.plot(wav_arima_ann_recon_df, color='black', label='WAV-ARIMA-ANN')
plt.plot(wav_ann_ann_recon_df, color='orange', label='WAV-ANN-ANN')
plt.legend(loc='upper left', prop={'size':24})
plt.title('Detail Forecasts', fontsize=30)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Dollars per Million Btu', fontsize=20)

plt.close('all')
plt.plot(data_pad[627:725])
plt.plot(wav_ann_ann_recon[627:725])
plt.plot(wav_ann_cD3cD1nn[627:725])
plt.plot(wav_ann_cD3cD2nn[627:725])
plt.plot(wav_ann_cD2cD1nn[627:725])
plt.plot(wav_ann_cD3nn[627:725])
plt.plot(wav_ann_cD2nn[627:725])
plt.plot(wav_ann_cD1nn[627:725])


