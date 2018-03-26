# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:53:54 2017

@author: Samuel
"""

"""Neural Network code to clean up other scripts that depend on it"""

def NeuralNet_ts_predictions(train_data, test_data, num_lags, hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08):

    import sklearn
    from sklearn.neural_network import MLPRegressor
    from sklearn import preprocessing
    
    #Create data frame to shift
    train_data_df = pd.DataFrame(train_data)
    test_data_df = pd.DataFrame(test_data)
    
    #Get features from lagged time series
    train_data_np = np.zeros((len(train_data),num_lags+1))
    train_data_df_combined = pd.DataFrame(train_data_np)
    train_data_df_combined.ix[:,0] = train_data_df
    test_data_np = np.zeros((len(test_data),num_lags+1))
    test_data_df_combined = pd.DataFrame(test_data_np)
    test_data_df_combined.ix[:,0] = test_data_df
    
    for i in range(1, num_lags+1):
        train_data_df_combined.ix[:,i] = train_data_df.shift(-i)  
        test_data_df_combined.ix[:,i] = test_data_df.shift(-i)

    #Drop rows with NANs
    train_data_combined_clean = train_data_df_combined.dropna()  
    test_data_combined_clean = test_data_df_combined.dropna()
    
    #Standardize the input data
    train_data_combined_standardized = preprocessing.scale(train_data_combined_clean)
    test_data_combined_standardized = preprocessing.scale(test_data_combined_clean)    
    
    #Target variable
    train_target = train_data_combined_standardized[:,-1]
    test_target = test_data_combined_standardized[:,-1]
    
    #Build the network model
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    
    #Fit the model to the training data
    mlp.fit(train_data_combined_standardized[:,:-1], train_target)
    
    #Predictions
    preds = mlp.predict(test_data_combined_standardized[:,:-1])
    
    #Evaluations of standardized predictions
    MSE_std = ((preds - test_target)**2).mean()
    RMSE_std = math.sqrt(MSE_std)
    print(MSE_std, RMSE_std, mlp.score(test_data_combined_standardized[:,:-1], test_target))
    
    #De-standardize predictions
    preds_unstd = preds * test_data_combined_clean.iloc[:,-1].std() + test_data_combined_clean.iloc[:,-1].mean()
    
    #Evaluations of unstandardized predictions
    MSE = ((preds_unstd - test_data_combined_clean.iloc[:,-1])**2).mean()
    RMSE = math.sqrt(MSE)
    print(MSE, RMSE)
    
    return preds, preds_unstd, MSE_std, RMSE_std, MSE, RMSE