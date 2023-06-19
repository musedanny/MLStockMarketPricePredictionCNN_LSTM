#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project
# 
# ## Stock Market Prices Prediction Using Machine Learning Techniques
# ### Work done by :
# ### Naomie Umwiza ID: 100662
# ### Denise Ahishakiye ID: 100636
# ### Daniel Musengimana ID: 100642

# ### Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten


# ### Load and concatenate data

# In[2]:


data_dir = '/Users/admin/Desktop/BIG_DATA/MachineLearning/FinalProject/StockMarketData'
data_files = glob.glob(data_dir + "/*.csv")
data_list = [pd.read_csv(file, index_col=None, header=0) for file in data_files]
combined_data = pd.concat(data_list, axis=0, ignore_index=True)

# Convert 'Close' to numeric and handle NaNs
combined_data['Close'] = pd.to_numeric(combined_data['Close'], errors='coerce')
combined_data.dropna(subset=['Close'], inplace=True)

# Handle missing values
combined_data.reset_index(drop=True, inplace=True)
combined_data.fillna(combined_data.mean(), inplace=True)


# In[3]:


combined_data.head(10)


# ## Data visualization

# ### Calculate the correlation matrix

# In[4]:


import seaborn as sns


correlation_matrix = combined_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[5]:


combined_data.shape


# ## Data Preprocessing

# In[6]:



# Get columns with missing values
columns_with_missing_values = combined_data.columns[combined_data.isnull().any()]

# Count the number of missing values in these columns
missing_values = combined_data[columns_with_missing_values].isnull().sum()

# Print the number of missing values in each of these columns
print("\nMissing values for null:")
print(missing_values)

# Identify missing values
missing_values_mask = combined_data.isnull()

# Count the number of missing values in each column
missing_values_count = missing_values_mask.sum()

# Check for missing values using isna()
nan_mask = combined_data.isna()

# Count the number of NaN values in these columns
nan_values = combined_data[columns_with_missing_values].isna().sum()

# Print the number of NaN values in each of these columns
print("\nMissing values for nan:")
print(nan_values)


# ### Removing unnecessary columns, columns to close to close price we will consider below data

# In[7]:


dataset=combined_data[['Symbol','Open', 'High', 'Low', 'Last','Close']]
dataset.head()


# In[8]:


dataset.describe()


# ### Filter the dataset for the top 5 Stocks:

# In[9]:


# Assuming 'performance_metric' column represents the performance of each Stocks
Stocks_performance = dataset.groupby('Symbol')['Close'].mean()
# Assuming 'performance_metric' column represents the performance of each Stocks
Stocks_performance = dataset.groupby('Symbol')['Close'].mean()
top_Stocks = Stocks_performance.nlargest(5).index
selected_Stocks_data = combined_data[dataset['Symbol'].isin(top_Stocks)]


# In[10]:


selected_Stocks_data.head()


# ### Data visualization for the to 5 performing Stocks from 2002-02-18 to 2021-04-30

# In[11]:


# Convert the date column to pandas datetime format
selected_Stocks_data['Date'] = pd.to_datetime(selected_Stocks_data['Date'])

plt.figure(figsize=(12, 6))

# Iterate over each Stocks symbol
for symbol in selected_Stocks_data['Symbol'].unique():
    Stocks_data = selected_Stocks_data[selected_Stocks_data['Symbol'] == symbol]

    # Resample the data to quarterly frequency
    quarterly_returns = Stocks_data.set_index('Date').resample('Q')['Close'].mean()

    # Fill missing quarters with NaN
    quarters = pd.date_range(start=quarterly_returns.index.min(), end=quarterly_returns.index.max(), freq='Q')
    quarterly_returns = quarterly_returns.reindex(quarters)

    # Plotting quarterly average daily returns
    plt.plot(quarterly_returns.index, quarterly_returns.values,marker='o', linestyle='-', label=symbol)

    # Add vertical lines at the boundaries of each quarter
    for quarter in quarters[1:]:
        plt.axvline(quarter, color='gray', linestyle='--', linewidth=0.5)

plt.title('Quarterly Average Daily Returns of Top Performing Stocks')
plt.xlabel('Date')
plt.ylabel('Average Daily Return')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# ### Create a scaler for our dataset exclude the symbol column

# In[12]:


dataset_final=dataset[['Open', 'High', 'Low', 'Last','Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset_final)

close_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the Close price
scaled_close = close_scaler.fit_transform(dataset_final[['Close']])

scaled_data[:, 3] = scaled_close.flatten()


# ### Prepare data for model

# In[13]:


X = []
Y = []
window_size = 100

for i in range(window_size, len(scaled_data) - window_size - 1):
    window_data = scaled_data[i-window_size:i]
    next_day_close_price = scaled_data[i + window_size, 3]  # Close price is at index 3
    X.append(window_data)
    Y.append(next_day_close_price)


# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)


# ### Split data into training and testing sets

# In[14]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Reshape data for the model
train_X = train_X.reshape(train_X.shape[0], 1, window_size, 5)
test_X = test_X.reshape(test_X.shape[0], 1, window_size, 5)
print('train_X shape: ', train_X.shape)
print('test_X shape: ', test_X.shape)


# In[15]:


# Define the model
def create_model():
    model = tf.keras.Sequential()
    
    # CNN layers
    #model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, 100, 1)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, window_size, 5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layers
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
    model.add(Dropout(0.5))
    
    # Dense layer
    model.add(Dense(1, activation='linear'))
   
    # Define the learning rate
    learning_rate = 0.001

    # Define the optimizer and specify the learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    return model


# In[16]:


# Call the function to create the model
model = create_model()

# Model training
history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=100, batch_size=60, verbose=1, shuffle=True)


# ### Save and reload model

# In[18]:


model.save("model_lstm_cnn_final_03.h5")
loaded_model = tf.keras.models.load_model("./model_lstm_cnn_final_03.h5")


# ###  Print the model summary

# In[19]:


model.summary()


# ### Model Evaluation (Calculate RMSE, MAE, MSE )

# In[22]:


test_loss, test_mse, test_mae = model.evaluate(test_X, test_Y)

# Calculate RMSE
rmse = np.sqrt(test_mse)
# print('Test RMSE: ', rmse)

test_loss_percentage = test_loss * 100
test_mse_percentage = test_mse * 100
test_mae_percentage = test_mae * 100
rmse_percentage = rmse * 100

print('Test Loss (test_loss) Score: {:.2f}%'.format(test_loss_percentage))
print('Test MSE (test_mse) Score: {:.2f}%'.format(test_mse_percentage))
print('Test MAE (test_mae) Score: {:.2f}%'.format(test_mae_percentage))
print('Test RMSE (test_mae) Score: {:.2f}%'.format(rmse_percentage))


# ### RE-Evaluate model  Calculate Accuracy with R-squared (R2) score

# In[23]:


from sklearn.metrics import r2_score

r2 = r2_score(test_Y_rescaled, predictions_rescaled)
r2_percentage = r2 * 100
print('R-squared (R2) Score: {:.2f}%'.format(r2_percentage))


# ### Make predictions

# In[25]:


predictions = model.predict(test_X)


# Rescale test_Y and predictions to original range using close_scaler
test_Y_rescaled = close_scaler.inverse_transform(test_Y.reshape(-1, 1))
predictions_rescaled = close_scaler.inverse_transform(predictions.reshape(-1, 1))


# ### Reshape test_Y and predictions to be one-dimensional

# In[27]:


test_Y = test_Y.reshape(-1)
predictions = predictions.reshape(-1)


# ### Rescale test_Y and predictions to original range

# In[30]:


test_Y_rescaled = close_scaler.inverse_transform(test_Y.reshape(-1, 1))
predictions_rescaled = close_scaler.inverse_transform(predictions.reshape(-1, 1))


# ### Visualize Prediction vs Original Data prices over the Time

# In[35]:


from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(12, 6))
plt.plot(test_Y_rescaled, 'b', label='Original Price')  # Close price
plt.plot(predictions_rescaled, 'r', label='Predicted Price')  # Close price

# Format the x-axis ticks with the desired date format
date_format = mdates.DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# ###  Visualize Predicted prices Then  Original Data prices over the Time

# In[33]:



test_Y_rescaled = close_scaler.inverse_transform(test_Y.reshape(-1, 1))
predictions_rescaled = close_scaler.inverse_transform(predictions.reshape(-1, 1))

# Plot the original price and predicted price
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot the original price in the upper subplot
ax1.plot(test_Y_rescaled, 'b', label='Original Price')  # Close price
ax1.set_ylabel('Price')
ax1.legend()

# Plot the predicted price in the lower subplot
ax2.plot(predictions_rescaled, 'r', label='Predicted Price')  # Close price
ax2.set_xlabel('Time')
ax2.set_ylabel('Price')
ax2.legend()

# Format the x-axis ticks with the desired date format
date_format = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()

plt.show()


# In[ ]:




