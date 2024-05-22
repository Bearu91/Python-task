import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Update these paths to the correct file locations on your machine
order_items_path = r'C:\Users\pc\Desktop\python test\S_Data\order_items.csv'
orders_path = r'C:\Users\pc\Desktop\python test\S_Data\orders.csv'
product_category_name_translation_path = r'C:\Users\pc\Desktop\python test\S_Data\product_category_name_translation.csv'
products_path = r'C:\Users\pc\Desktop\python test\S_Data\products.csv'

# Load the data
order_items = pd.read_csv(order_items_path)
orders = pd.read_csv(orders_path)
product_category_name_translation = pd.read_csv(product_category_name_translation_path)
products = pd.read_csv(products_path)

# Merge datasets as necessary to create a comprehensive dataset
merged_data = orders.merge(order_items, on='order_id', how='inner')
merged_data = merged_data.merge(products, on='product_id', how='inner')
merged_data = merged_data.merge(product_category_name_translation, on='product_category_name', how='inner')

# Convert date columns to datetime
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

# Aggregate data to daily level
daily_data = merged_data.groupby(['order_purchase_timestamp', 'product_category_name_english']).agg(
    {'price': 'sum'}).reset_index()
daily_data.rename(
    columns={'order_purchase_timestamp': 'date', 'product_category_name_english': 'product_group', 'price': 'sales'},
    inplace=True)

# Feature engineering for machine learning
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
daily_data['month'] = daily_data['date'].dt.month

# Split data into training and test sets
train_data, test_data = train_test_split(daily_data, test_size=0.2, shuffle=False)

# Set the date as index for SARIMA and ARIMA models and specify frequency
train_data.set_index('date', inplace=True)
train_data.index = train_data.index.to_period('D')
test_data.set_index('date', inplace=True)
test_data.index = test_data.index.to_period('D')


# Function to forecast using SARIMA
def sarima_forecast(train_df, test_df, product_grp):
    train_group = train_df[train_df['product_group'] == product_grp]
    test_group = test_df[test_df['product_group'] == product_grp]

    model = SARIMAX(train_group['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), freq='D')
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test_group))

    return forecast, model_fit


# Function to forecast using ARIMA
def arima_forecast(train_df, test_df, product_grp):
    train_group = train_df[train_df['product_group'] == product_grp]
    test_group = test_df[test_df['product_group'] == product_grp]

    model = ARIMA(train_group['sales'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_group))

    return forecast, model_fit


# Example forecast for a product group using SARIMA and ARIMA
example_product_group = 'electronics'
sarima_forecast_result, sarima_model_fit = sarima_forecast(train_data, test_data, example_product_group)
arima_forecast_result, arima_model_fit = arima_forecast(train_data, test_data, example_product_group)

# Prepare feature matrix for machine learning
X_train = train_data[['day_of_week', 'month']]
y_train = train_data['sales']
X_test = test_data[['day_of_week', 'month']]
y_test = test_data['sales']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

# Forecast for test period using Random Forest
rf_forecast = rf_model.predict(X_test_scaled)
mae_rf = mean_absolute_error(y_test, rf_forecast)

print(f'Mean Absolute Error (Random Forest): {mae_rf}')


# Using average demand of similar products as a baseline for new products
def forecast_new_product(train_df, product_grp):
    similar_products = train_df[train_df['product_group'] == product_grp]
    average_demand = similar_products['sales'].mean()
    forecast = [average_demand] * 14  # Forecast next 14 days

    return forecast


# Example forecast for a new product group
new_product_group = 'new_electronics'
new_product_forecast = forecast_new_product(train_data, new_product_group)

# Evaluate SARIMA and ARIMA
sarima_pred = sarima_forecast_result
arima_pred = arima_forecast_result
sarima_mae = mean_absolute_error(test_data[test_data['product_group'] == example_product_group]['sales'], sarima_pred)
arima_mae = mean_absolute_error(test_data[test_data['product_group'] == example_product_group]['sales'], arima_pred)

# Evaluate Random Forest
rf_pred = rf_model.predict(X_test_scaled)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f'SARIMA MAE: {sarima_mae}')
print(f'ARIMA MAE: {arima_mae}')
print(f'Random Forest MAE: {rf_mae}')

# Save models
with open('sarima_model.pkl', 'wb') as sarima_output_file:
    pickle.dump(sarima_model_fit, sarima_output_file)
with open('arima_model.pkl', 'wb') as arima_output_file:
    pickle.dump(arima_model_fit, arima_output_file)
with open('rf_model.pkl', 'wb') as rf_output_file:
    pickle.dump(rf_model, rf_output_file)


# Function to generate forecasts
def generate_forecast(product_grp, model_type='sarima'):
    if model_type == 'sarima':
        with open('sarima_model.pkl', 'rb') as sarima_input_file:
            model = pickle.load(sarima_input_file)
        forecast = model.forecast(steps=14)
    elif model_type == 'arima':
        with open('arima_model.pkl', 'rb') as arima_input_file:
            model = pickle.load(arima_input_file)
        forecast = model.forecast(steps=14)
    else:
        with open('rf_model.pkl', 'rb') as rf_input_file:
            model = pickle.load(rf_input_file)
        forecast = model.predict(X_test_scaled)[:14]

    return forecast


# Example usage
print(generate_forecast('electronics', model_type='sarima'))
print(generate_forecast('electronics', model_type='arima'))
print(generate_forecast('electronics', model_type='random_forest'))
