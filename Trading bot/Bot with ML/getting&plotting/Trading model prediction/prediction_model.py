import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'TTE'

start = dt.datetime(2000, 1, 3)
end = dt.datetime(2008, 7, 25)

data = pdr.get_data_yahoo(company, start, end)
#data = web.DataReader(company, 'yahoo', start, end)

# Prepare Data                                                                                                                                              
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build The Model                                                                                                                                           
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(
    units=1))  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=40, batch_size=32)

''' Test The Model Accuracy on Existing Data '''

# Load Test Data                                                                                                                                            
test_start = dt.datetime(2016, 7, 4)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']))

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make Prediction on Test Data                                                                                                                              

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot The Test Prediction                                                                                                                                  
# plt.plot(actual_prices, color='black', label=f"Actuel {company} Price")                                                                                   
# plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")                                                                             
# plt.title(f"{company} Share Price")                                                                                                                       
# plt.xlabel('Time')                                                                                                                                        
# plt.ylabel(f'{company} Share Price')                                                                                                                      
# plt.xlim(1000,1300)                                                                                                                                       
# plt.legend()                                                                                                                                              
# plt.savefig('/Users/bastienscheunemann/Desktop/Trading strategy/Trading model prediction/prediction_test.png')                                            

# Load Data                                                                                                                                                 
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# Predict Next Day                                                                                                                                          
prediction_scaler = model.predict(real_data)
prediction = scaler.inverse_transform(prediction_scaler)
print(f"Prediction: {prediction}")

# Predict Next 10 Days                                                                                                                                      

prediction_list = [prediction]

for _ in range(9):
    real_data = np.append([], np.append(real_data[0][1:], [prediction_scaler]))
    real_data = [real_data]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction_scaler = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction_scaler)
    prediction_list = np.append(prediction_list, prediction)

print(prediction_list)

res0 = [[48.5, 47.746246, 47.92512, 48.029545, 48.08163, 48.101562, 48.10453, 48.100517, 48.095295, 48.091743],
        [48.5, 48.229744, 48.44657, 48.61145, 48.741524, 48.850826, 48.94908, 49.042343, 49.13397, 49.225704],
        [48.5,  47.660816, 47.801735, 47.870632, 47.89265, 47.888165, 47.87108, 47.8498, 47.82889, 47.810524],
        [48.5, 48.328163, 48.54617, 48.71937, 48.862667, 48.988266, 49.104675, 49.217136, 49.328518, 49.440212],
        [48.5, 48.400017, 48.60391, 48.75988, 48.885372, 48.99395, 49.094666, 49.192905, 49.29155, 49.391975],
        [48.5, 47.259117, 47.349766, 47.34827, 47.285484, 47.18751, 47.072983, 46.953705, 46.8365, 46.72479],
        [48.5,  48.127823, 48.33339, 48.478165, 48.58049, 48.656727, 48.719036, 48.77538, 48.830433, 48.886593],
        [48.5,  48.213646, 48.43059, 48.59759, 48.733204, 48.852306, 48.964916, 49.076874, 49.191135, 49.30884],
        [48.5, 47.801136, 47.96084, 48.04833, 48.08727, 48.09789, 48.094765, 48.087227, 48.080643, 48.077744],
        [48.5, 47.12917, 47.196426, 47.164375, 47.06617, 46.930794, 46.779285, 46.625034, 46.475636, 46.334686],
        [48.5, 48.456814, 48.700867, 48.89608, 49.058197, 49.200752, 49.333614, 49.463097, 49.5928, 49.72452],
        [48.5,  48.467014, 48.696064, 48.88204, 49.040333, 49.18366, 49.32083, 49.45724, 49.595764, 49.737667],
        [48.5,  46.961773, 47.021282, 46.972656, 46.85185, 46.689438, 46.50723, 46.318947, 46.13236, 45.951283],
        [48.5,  48.620785, 48.855095, 49.052334, 49.226093, 49.38708, 49.542507, 49.6967, 49.85197, 50.00942],
        [48.5,  47.674164, 47.824043, 47.890907, 47.90075, 47.87738, 47.838764, 47.79669, 47.75787, 47.72545],
        [48.5, 47.993813, 48.146984, 48.229103, 48.263905, 48.2711, 48.26446, 48.25259, 48.24046, 48.230618],
        [48.5,  47.43297, 47.556316, 47.597965, 47.584316, 47.538055, 47.47559, 47.407467, 47.339848, 47.27592],
        [48.5,  47.406662, 47.53271, 47.56908, 47.543056, 47.47908, 47.3954, 47.30404, 47.2121, 47.12332],
        [48.5,  48.62832, 48.846222, 49.024483, 49.1766, 49.31339, 49.442326, 49.568123, 49.69354, 49.820095],
        [48.5,  47.13349, 47.191166, 47.149384, 47.041714, 46.896976, 46.73609, 46.5726, 46.414433, 46.265614],
        [48.5,  48.236393, 48.431004, 48.56642, 48.662872, 48.737236, 48.80151, 48.86328, 48.926876, 48.99445],
        [48.5, 47.877274, 48.046486, 48.14682, 48.198944, 48.221153, 48.227116, 48.22595, 48.223175, 48.22187]]


def moyenne(t):

    S = 0

    for i in t:

        S+=i

    return S/len(t)


def moyenne_T(t):

    n = len(t[0])

    T = [0 for _ in range(n)]

    for i in range(n):

        t0 = [c[i] for c in t]

        T[i] = moyenne(t0)

    return T


x = [x+len(actual_prices) for x in range(10)]

plt.plot(actual_prices, color='black', label=f"Actuel {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
plt.plot(x, prediction_list)

#for i in range(len(res0[0])):
#    plt.plot(x, res0[i])

plt.plot(x, moyenne_T(res0), color='red', label='10 days average Price')
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.xlim(1300, 1360)
plt.ylim(45, 50)
plt.legend()
plt.savefig('/Users/bastienscheunemann/Desktop/Trading strategy/Trading model prediction/prediction_10days.png')
