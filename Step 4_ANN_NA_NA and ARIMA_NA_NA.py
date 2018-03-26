# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:58:25 2017

@author: Samuel
"""

"""ANN-N/A-N/A"""
MSE_arr = []
order = []

for j in range(1,11):
    preds_orig, preds_unstd_orig, MSE_std, RMSE_std, MSE, RMSE = NeuralNet_ts_predictions(data_ts1, test_ts1, num_lags=3, hidden_layer_sizes=(j,), activation='relu', random_state=42, solver='adam', max_iter=1000, tol=1e-10)
    MSE_arr = MSE_arr + [MSE]
    order = order + [j]
    print(min(MSE_arr), MSE_arr.index(min(MSE_arr)), order[MSE_arr.index(min(MSE_arr))])

#Minimum MSE when num_lags = 6 (i=6) and number of neurons in the hidden layer = 2 (j=2)
#MSE = 0.0471347462999
#RMSE = 0.21710538063319457

#According to paper, need 3 lags, 3 neurons in 1 hidden layer
preds_orig, preds_unstd_orig, MSE_std, RMSE_std, MSE, RMSE = NeuralNet_ts_predictions(data_ts1, test_ts1, num_lags=3, hidden_layer_sizes=(3,), random_state=42, solver='lbfgs', max_iter=1000, tol=1e-20)

MAPE_orig = (abs(preds_unstd_orig-np.ravel(data[627:725]))/np.ravel(data[627:725])).sum()/len(preds_unstd_orig)
print(MAPE_orig)
#MAPE = 0.0368642736914

"""ARIMA-N/A-N/A"""
import statsmodels.api as sm
arima_orig_train_mod = sm.tsa.statespace.SARIMAX(data[0:624],order=(0,1,1))
arima_orig_train_res = arima_orig_train_mod.fit(maxiter=100)

arima_orig = sm.tsa.SARIMAX(data, order=(0,1,1))
arima_orig_res = arima_orig.filter(arima_orig_train_res.params)

arima_orig_preds = arima_orig_res.predict(start=624,end=724)

MSE_arima_orig = ((arima_orig_preds[3:]-np.ravel(data[627:725]))**2).mean()
RMSE_arima_orig = math.sqrt(MSE_arima_orig)
MAPE_arima_orig = (abs(arima_orig_preds[3:]-np.ravel(data[627:725]))/np.ravel(data[627:725])).sum()/len(arima_orig_preds[3:])
print(MSE_arima_orig, RMSE_arima_orig, MAPE_arima_orig)
#MSE = 0.019654105103565883
#RMSE = 0.14019309934360494
#MAPE = 0.037738759706018046

"""Plot"""
preds_unstd_orig_df = pd.DataFrame(preds_unstd_orig, index=data[627:725].index)
plt.plot(data[627:725], label='Actual')
plt.plot(arima_orig_preds[3:], label='ARIMA')
plt.plot(preds_unstd_orig_df, label='ANN')
plt.legend(loc='upper left', prop={'size':24})
plt.title('Actual vs. ARIMA vs. ANN', fontsize=30)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Dollars per Million Btu', fontsize=20)


