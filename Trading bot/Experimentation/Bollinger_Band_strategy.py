import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

with open('liste_valeur_BB', 'r') as f:
    _ = f.readline()
    T = f.readlines()
    L = []
    for i in range(len(T)):
        T[i] = T[i].strip().split('\t')
        L.append(T[i][0])


def Sharpe_Ratio(S):
    plt.style.use('ggplot')

    data = yf.download(S, start='2018-01-01', end='2019-01-01')

    data.index = pd.to_datetime(data.index, True)

    data['MA'] = data.Close.rolling(20).mean()
    data['Variance'] = data.Close.rolling(20).std().mean()

    data['Upper_Band'] = data['MA'] + 2 * data['Variance']
    data['Lower_Band'] = data['MA'] - 2 * data['Variance']

    data['trading_signal'] = np.where(abs(data['Close'] - 2*data['Variance'])
                                      < data['MA'], 1, -1)

    data['returns'] = data.Close.pct_change()
    data['Strategy_returns'] = data.returns * data.trading_signal.shift(1)
    data['cumulative_returns'] = (data.Strategy_returns + 1).cumprod()
    #data[['Strategy_returns']].plot()
    #plt.show()

    trading_days = 252

    annuals_returns = ((1 + data.returns.mean() ** 1/2) - 1) * 100

    risk_free_rate = 0.06
    daily_risk_free_return = risk_free_rate / trading_days

    excess_daily_returns = data.returns - daily_risk_free_return

    sharpe_ratio = (excess_daily_returns.mean() /
                    excess_daily_returns.std()) * np.sqrt(trading_days)

    return sharpe_ratio, annuals_returns


plt.clf()
x_list = []
y_list = []
for l in L :
    x_list.append(Sharpe_Ratio(l)[0])
    y_list.append(Sharpe_Ratio(l)[1])
    plt.plot(x_list, y_list, '.', )
plt.savefig('graph_BB.png')

