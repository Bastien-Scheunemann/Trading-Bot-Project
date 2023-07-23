import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

with open('liste_valeur_MACD', 'r') as f:
    _ = f.readline()
    T = f.readlines()
    L = []
    for i in range(len(T)):
        T[i] = T[i].strip().split('\t')
        L.append(T[i][0])


def Sharpe_Ratio(S):
    plt.style.use('ggplot')

    data = yf.download(S, start='2018-01-01',
                       end='2020-01-01')
#=datetime.datetime.today().strftime('%Y-%m-%d')
    data.index = pd.to_datetime(data.index, True)

    # data['Close'].plot()
    # plt.show()

    data['12d_EMA'] = data.Close.ewm(span=12).mean()
    data['26d_EMA'] = data.Close.ewm(span=26).mean()

    # data[['Close', '12d_EMA', '26d_EMA']].plot()

    # plt.show()

    data['MACD'] = data['26d_EMA'] - data['12d_EMA']
    data['Signal'] = data.MACD.ewm(span=9).mean()

    # data[['MACD', 'Signal']].plot()
    # plt.show()

    data['trading_signal'] = np.where(data['MACD'] > data['Signal'], 1, -1)
    data['returns'] = data.Close.pct_change()
    data['Strategy_returns'] = data.returns * data.trading_signal.shift(1)
    cumulative_returns = (data.Strategy_returns + 1).cumprod()

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
plt.savefig('graph_MACD.png')


"""
with open('liste_valeur_MACD', 'w') as f:
    V = []
    for i in range(len(L)):
        if Sharpe_Ratio(L[i]) > 0:
            V.append([Sharpe_Ratio(L[i]), L[i]])
    V.sort()
    i = 1
    for v in V:
        f.write(str(i) + ' ' + str(v[1]) + ' ' + str(v[0]) + '\n')
        i += 1
"""

