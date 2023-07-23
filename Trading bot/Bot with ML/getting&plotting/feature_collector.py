import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Experimentation.functions import *
from sklearn import *

# Load our CSV Data

data = pd.read_csv('DATA/EURUSD_Candlestick_1_h_ASK_01.01.2019-28.03.2020.csv')

data.columns = ['Date', 'open', 'high', 'low', 'close', 'volume']

data = data.set_index(pd.to_datetime(data.Date))

data = data[['open', 'high', 'low', 'close', 'volume']]

prices = data.drop_duplicates(keep=False)

# Create lists for each periods required by our functions

momentumKey = [3, 4, 5, 8, 9, 10]
stochasticKey = [3, 4, 5, 8, 9, 10]
williamsKey = [6, 7, 8, 9, 10]
procKey = [12, 13, 14, 15]
wadlKey = [15]
adoscKey = [2, 3, 4, 5]
cciKey = [15]
bollingerKey = [15]
heikenashiKey = [15]
paverageKey = [2]
slopeKey = [3, 4, 5, 10, 20, 30]
#fourierKey = [10, 20, 30]
#sineKey = [5, 6]

keylist = [momentumKey, stochasticKey, williamsKey, procKey, wadlKey, adoscKey, cciKey, bollingerKey,
           heikenashiKey, paverageKey, slopeKey]

# Calculate all of the features

momentumDict = momentum(prices, momentumKey)
print('1')
stochasticDict = stochastic(prices, stochasticKey)
print('2')
williamsDict = williams(prices, williamsKey)
print('3')
procDict = proc(prices, procKey)
print('4')
wadlDict = wadl(prices, wadlKey)
print('5')
adoscDict = adosc(prices, adoscKey)
print('6')
cciDict = cci(prices, cciKey)
print('8')
bollingerDict = bollinger(prices, bollingerKey, 2)
print('9')

hkaprices = prices.copy()
hkaprices['Symbol'] = 'SYMB'

HKA = OHLCresample(hkaprices, '15H')

heikenDict = heikenashi(HKA, heikenashiKey)
print('10')
paverageDict = paverage(prices, paverageKey)
print('11')
slopeDict = slopes(prices, slopeKey)
print('12')


# Create list of dictionnaries

dictlist = [momentumDict.close, stochasticDict.close, williamsDict.close, procDict.proc, wadlDict.wadl,
            adoscDict.AD, cciDict.cci, bollingerDict.bands, heikenDict.candles, paverageDict.avs,
            slopeDict.slope]

# List of 'base' column names

colFeat = ['momentum', 'stoch', 'will', 'proc', 'wadl', 'adosc', 'cci', 'bollinger', 'heiken', 'paverage', 'slope']

# Populate the MASTERFRAME

masterFrame = pd.DataFrame(index=prices.index)

for i in range(0, len(keylist)):

    for j in keylist[i]:

        for k in list(dictlist[i][j]):
            colID = colFeat[i] + str(j) + str(k)

            masterFrame[colID] = dictlist[i][j][k]

threshold = round(0.7 * len(masterFrame))

masterFrame[['open', 'high', 'low', 'close']] = prices[['open', 'high', 'low', 'close']]

# Heiken Ashi is resampled ==> empty data in between

masterFrame.heiken15open = masterFrame.heiken15open.fillna(method='bfill')
masterFrame.heiken15high = masterFrame.heiken15high.fillna(method='bfill')
masterFrame.heiken15low = masterFrame.heiken15low.fillna(method='bfill')
masterFrame.heiken15close = masterFrame.heiken15close.fillna(method='bfill')

# Drop columns that have 30% or more NAN data

masterFrameCleaned = masterFrame.copy()

masterFrameCleaned = masterFrameCleaned.dropna(axis=1, thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)
masterFrameCleaned['prices'] = masterFrameCleaned.close.pct_change()

masterFrameCleaned.to_csv('DATA/masterframe.csv')

print('Completed Feature Calculations')

masterFrameCleaned['prices'] = masterFrameCleaned.close.pct_change()


pr = masterFrameCleaned['prices']
pr[pr > 0] = 1
pr[pr < 0] = -1
pr[pr == 0] = 0

print(pr)

#print("masterframe: {}".format(masterFrameCleaned['prices']))

#print(" Keys of masterFrameCleaned: {}".format(masterFrameCleaned.keys()))


"""
X_train, X_test, y_train, y_test = train_test_split(
    masterFrameCleaned
    masterFrameCleaned['prices'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
"""