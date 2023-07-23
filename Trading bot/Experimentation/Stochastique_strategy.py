import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


with open('liste_valeur_Stochastique.rtf', 'r') as f:
    _ = f.readline()
    T = f.readlines()
    L = []
    for i in range(len(T)):
        T[i] = T[i].strip().split(' ')
        L.append(T[i][1])


def Sharpe_Ratio(S):
    data = yf.download(S, start= '2018-01-01',
                       end= datetime.datetime.today().strftime('%Y-%m-%d'))

    data.index = pd.to_datetime(data.index, True)

    data['%K'] = (data['Close'] - data['Low'])/(data['High'] - data['Low'])
    data['%D'] = data['%K'].ewm(span=9).mean()

    data['Trading_Signal'] = np.where(data['%K'] > data['%D'], 1, -1)
    data['returns'] = data.Close.pct_change()
    data['Strategy_returns'] = data.returns * data.Trading_Signal.shift(1)
    cumulative_returns = (data.Strategy_returns + 1).cumprod()
    #cumulative_returns.plot()
    #plt.show()

    trading_days = 0.5

    annuals_returns = ((1+ data.returns.mean() ** trading_days) - 1) * 100

    risk_free_rate = 0.06
    daily_risk_free_return = risk_free_rate / trading_days
    excess_daily_returns = data.returns - daily_risk_free_return

    sharpe_ratio = (excess_daily_returns.mean() /
                    excess_daily_returns.std() * np.sqrt(trading_days))

    return sharpe_ratio, annuals_returns



plt.clf()
x_list = []
y_list = []
for l in L :
    x_list.append(Sharpe_Ratio(l)[0])
    y_list.append(Sharpe_Ratio(l)[1])
    plt.plot(x_list, y_list, '.', )
plt.savefig('graph_Stochastique.png')


"""
with open('liste_valeur_Stochastique.rtf', 'w') as f:
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