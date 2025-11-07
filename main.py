#Needed imports for NN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt 
'''numpy: to use arrays
matplotlib for graphs
pandas: for data manipulation
and MinMaxScaler to normalize between 0 and 1 (under)'''


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#...
#...
#PREPARING MODEL DATA 

# To load the data we put the Company, the start and end of the training data
company = 'NVDA'
start = dt.datetime(2016,1,1)
end = dt.datetime(2024,11,1)

#literraly importing the data
data = yf.download(company, start=start, end=end)

# Preparing  this data
scaler = MinMaxScaler(feature_range=(0,1)) #creating a scaler that will normalize values within 0-1
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))#we want to scale the data based on closing prices (whole point of this NN)

prediction_days = 60
x_train = []#contains sequence of 60 consequtive days (closing prices)
y_train = []#contian the price of prediction_day + 1 (aka what we want to predict)

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])


#we want to convert lists to numpy arrays (easier to train: or necessary idk)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#...
#...

# MODEL BEING ACTUALLY BUILT
model = Sequential() #sequential is when layers are stacked one after the other (crucial because we want to predict based on an ordered list of closing prices)
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))#single neuron that outputs the predicted price
# we used 50 memory units and randomly dropped 20% during training 


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)  #Increased epochs

#Test the model
test_start = dt.datetime(2024,11,1)
test_end = dt.datetime.now()
test_data = yf.download(company, start=test_start, end=test_end)

actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)# Combines training and test data

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)#normalize using scaler

#Making predictions on test data
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)#THIS CONVERTS NORMALIZED PREDICTIONS BACK TO ACTUAL PRICES

#THIS IS JUST BASIC PLOTING USING MATPLOTLIB LIBRARY
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color='black', label='Actual Price', linewidth=2)
plt.plot(predicted_prices, color='green', label='Predicted Price', linewidth=2)
plt.title(f"{company} Share Price Prediction")
plt.xlabel('Days')
plt.ylabel(f'{company} Share Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Predicting next day
# Get the most recent 60 days of data
latest_data = yf.download(company, period='3mo')  # Get recent data
last_60_days = latest_data['Close'].values[-prediction_days:]

#Scale the data
last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

#Prepare for prediction
real_data = np.array([last_60_days_scaled[:, 0]])
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

#STEP THAT PREDICTS (Unkown response: what i mean is that you need to wait and see if the pred is right)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)


# Just terminal printing of the prediction once the model is trained localy
print(f"\nPrediction for next trading day: ${prediction[0][0]:.2f}")
print(f"Current price: ${latest_data['Close'].values[-1]:.2f}")

print(f"Predicted change: ${prediction[0][0] - latest_data['Close'].values[-1]:.2f} ({((prediction[0][0] / latest_data['Close'].values[-1]) - 1) * 100:.2f}%)")
