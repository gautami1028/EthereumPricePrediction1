import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os.path
import pickle

# Function to fetch historical Ethereum price data from CoinGecko API
def fetch_ethereum_data():
    # Make a request to the CoinGecko API to fetch historical Ethereum price data
    response = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=365')
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON data
        ethereum_data = response.json()
        
        # Extract relevant data from the response
        prices = ethereum_data['prices']
        ethereum_df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
        
        # Convert timestamp to datetime
        ethereum_df['Timestamp'] = pd.to_datetime(ethereum_df['Timestamp'], unit='ms')
        
        return ethereum_df
    else:
        # If the request failed, return None
        print("Error in API")
        return None

# Function to train the model
def train_model(X, y):
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler.feature_names_in_ = ['Year', 'Month', 'Day', 'Weekday']

    # Hyperparameter tuning for Random Forest
    rf_param_grid = {'n_estimators': [100, 200, 300],
                     'max_depth': [None, 10, 20, 30],
                     'min_samples_split': [2, 5, 10]}
    rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5)
    rf_grid_search.fit(X_scaled, y)
    best_rf_model = rf_grid_search.best_estimator_

    # Hyperparameter tuning for Extra Trees
    et_param_grid = {'n_estimators': [100, 200, 300],
                     'max_depth': [None, 10, 20, 30],
                     'min_samples_split': [2, 5, 10]}
    et_grid_search = GridSearchCV(ExtraTreesRegressor(), et_param_grid, cv=5)
    et_grid_search.fit(X_scaled, y)
    best_et_model = et_grid_search.best_estimator_
    
    return best_rf_model, best_et_model, scaler

# Function to check if the model needs to be retrained
def check_if_retrain_needed(ethereum_data):
    # Check if the model file exists
    if not os.path.exists("trained_model.pkl"):
        return True
    
    # Check the last modification time of the model file
    last_modified_time = datetime.fromtimestamp(os.path.getmtime("trained_model.pkl"))
    
    # Check if the last training time is more than a day ago
    if datetime.now() - last_modified_time > timedelta(days=1):
        return True
    else:
        return False

# Load historical Ethereum price data from the CoinGecko API
ethereum_data = fetch_ethereum_data()

if ethereum_data is not None and check_if_retrain_needed(ethereum_data):
    
    ethereum_data['Year'] = ethereum_data['Timestamp'].dt.year
    ethereum_data['Month'] = ethereum_data['Timestamp'].dt.month
    ethereum_data['Day'] = ethereum_data['Timestamp'].dt.day
    ethereum_data['Weekday'] = ethereum_data['Timestamp'].dt.weekday
        
    # Split data into features (X) and target variable (y)
    X = ethereum_data[['Year', 'Month', 'Day', 'Weekday']]  
    y = ethereum_data['Price']  
    
    # Train the model
    best_rf_model, best_et_model, scaler = train_model(X, y)
    
    # Save the trained model and scaler
    
    with open("trained_model.pkl", "wb") as f:
        pickle.dump((best_rf_model, best_et_model, scaler), f)
elif ethereum_data is not None and not check_if_retrain_needed(ethereum_data):
    print("Model is already up to date, no need to retrain.")
else:
    print("Failed to fetch Ethereum data from the CoinGecko API.")

# Load the pre-trained model and scaler
if os.path.exists("trained_model.pkl"):
    with open("trained_model.pkl", "rb") as f:
        best_rf_model, best_et_model, scaler = pickle.load(f)

# Function to predict Ethereum price for future dates using best Random Forest model
def predict_prices_for_future_random_forest(date):
    # Preprocess input date provided by the user
    date = pd.to_datetime(date)
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()

    # Feature scaling for prediction
    scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

    # Make prediction
    predicted_price = best_rf_model.predict(scaled_features_for_date)[0]

    # Return tuple of predicted price and default value for predicted high and low
    return predicted_price, predicted_price+100, predicted_price-100

# Function to predict Ethereum price for future dates using best Extra Trees model
def predict_prices_for_future_extra_trees(date):
    # Preprocess input date provided by the user
    date = pd.to_datetime(date)
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()

    # Feature scaling for prediction
    scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

    # Make prediction
    predicted_price = best_et_model.predict(scaled_features_for_date)[0]

    return predicted_price, predicted_price+100, predicted_price-100

#     # Calculate MAE, MSE, RMSE, and R2 for Random Forest
# rf_mae = mean_absolute_error(y, rf_predictions)
# rf_mse = mean_squared_error(y, rf_predictions)
# rf_rmse = mean_squared_error(y, rf_predictions, squared=False)
# rf_r2 = r2_score(y, rf_predictions)

# # Calculate MAE, MSE, RMSE, and R2 for Extra Trees

# et_mae = mean_absolute_error(y, et_predictions)
# et_mse = mean_squared_error(y, et_predictions)
# et_rmse = mean_squared_error(y, et_predictions, squared=False)
# et_r2 = r2_score(y, et_predictions)

# print("Random Forest Metrics:")
# print("MAE:", rf_mae)
# print("MSE:", rf_mse)
# print("RMSE:", rf_rmse)
# print("R-squared:", rf_r2)

# print("\nExtra Trees Metrics:")
# print("MAE:", et_mae)
# print("MSE:", et_mse)
# print("RMSE:", et_rmse)
# print("R-squared:", et_r2)



# import pandas as pd
# import requests
# from datetime import datetime
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Function to fetch historical Ethereum price data from CoinGecko API
# def fetch_ethereum_data():
#     # Make a request to the CoinGecko API to fetch historical Ethereum price data
#     response = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=365')
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Parse the response JSON data
#         ethereum_data = response.json()
        
#         # Extract relevant data from the response
#         prices = ethereum_data['prices']
#         ethereum_df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
        
#         # Convert timestamp to datetime
#         ethereum_df['Timestamp'] = pd.to_datetime(ethereum_df['Timestamp'], unit='ms')
        
#         return ethereum_df
#     else:
#         # If the request failed, return None
#         print("Error in api")
#         return None

# # Load historical Ethereum price data from the CoinGecko API
# ethereum_data = fetch_ethereum_data()

# if ethereum_data is not None:
#     # Data preprocessing
#     ethereum_data['Year'] = ethereum_data['Timestamp'].dt.year
#     ethereum_data['Month'] = ethereum_data['Timestamp'].dt.month
#     ethereum_data['Day'] = ethereum_data['Timestamp'].dt.day
#     ethereum_data['Weekday'] = ethereum_data['Timestamp'].dt.weekday
    
#     # Split data into features (X) and target variable (y)
#     X = ethereum_data[['Year', 'Month', 'Day', 'Weekday']]  # Features
#     y = ethereum_data['Price']  # Target variable

#     # Feature scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     scaler.feature_names_in_ = ['Year', 'Month', 'Day', 'Weekday']

#     # Hyperparameter tuning for Random Forest
#     rf_param_grid = {'n_estimators': [100, 200, 300],
#                      'max_depth': [None, 10, 20, 30],
#                      'min_samples_split': [2, 5, 10]}
#     rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5)
#     rf_grid_search.fit(X_scaled, y)
#     best_rf_model = rf_grid_search.best_estimator_

#     # Hyperparameter tuning for Extra Trees
#     et_param_grid = {'n_estimators': [100, 200, 300],
#                      'max_depth': [None, 10, 20, 30],
#                      'min_samples_split': [2, 5, 10]}
#     et_grid_search = GridSearchCV(ExtraTreesRegressor(), et_param_grid, cv=5)
#     et_grid_search.fit(X_scaled, y)
#     best_et_model = et_grid_search.best_estimator_

#     # Function to predict Ethereum price for future dates using best Random Forest model
#     def predict_prices_for_future_random_forest(date):
#         # Preprocess input date provided by the user
#         date = pd.to_datetime(date)
#         year = date.year
#         month = date.month
#         day = date.day
#         weekday = date.weekday()

#         # Feature scaling for prediction
#         scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

#         # Make prediction
#         predicted_price = best_rf_model.predict(scaled_features_for_date)[0]

#         # Return tuple of predicted price and default value for predicted high and low
#         return predicted_price, predicted_price+100, predicted_price-100

#     # Function to predict Ethereum price for future dates using best Extra Trees model
#     def predict_prices_for_future_extra_trees(date):
#         # Preprocess input date provided by the user
#         date = pd.to_datetime(date)
#         year = date.year
#         month = date.month
#         day = date.day
#         weekday = date.weekday()

#         # Feature scaling for prediction
#         scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

#         # Make prediction
#         predicted_price = best_et_model.predict(scaled_features_for_date)[0]

#         print( predicted_price)
        
#         return predicted_price, predicted_price+100, predicted_price-100
    
#     # checking for errors in model
    
#     # Make predictions using the best Random Forest model
#     rf_predictions = best_rf_model.predict(X_scaled)

#     # Make predictions using the best Extra Trees model
#     et_predictions = best_et_model.predict(X_scaled)

#     # Calculate MAE, MSE, RMSE, and R2 for Random Forest
#     rf_mae = mean_absolute_error(y, rf_predictions)
#     rf_mse = mean_squared_error(y, rf_predictions)
#     rf_rmse = mean_squared_error(y, rf_predictions, squared=False)
#     rf_r2 = r2_score(y, rf_predictions)

#     # Calculate MAE, MSE, RMSE, and R2 for Extra Trees
#     et_mae = mean_absolute_error(y, et_predictions)
#     et_mse = mean_squared_error(y, et_predictions)
#     et_rmse = mean_squared_error(y, et_predictions, squared=False)
#     et_r2 = r2_score(y, et_predictions)

#     print("Random Forest Metrics:")
#     print("MAE:", rf_mae)
#     print("MSE:", rf_mse)
#     print("RMSE:", rf_rmse)
#     print("R-squared:", rf_r2)

#     print("\nExtra Trees Metrics:")
#     print("MAE:", et_mae)
#     print("MSE:", et_mse)
#     print("RMSE:", et_rmse)
#     print("R-squared:", et_r2)
    
# else:
#     print("Failed to fetch Ethereum data from the CoinGecko API.")
