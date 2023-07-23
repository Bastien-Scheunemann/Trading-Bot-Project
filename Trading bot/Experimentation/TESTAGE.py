import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

with open('liste_valeur_MACD', 'r') as f:
    _ = f.readline()
    T = f.readlines()
    L = []
    for i in range(len(T)):
        T[i] = T[i].strip().split('\t')
        L.append(T[i][0])


def Sharpe_Ratio(S):
    plt.style.use('ggplot')
    data = yf.download(S, start='2016-01-01',
                       end='2019-01-01')

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
    cumulative_returns.plot(color='b')

    data['MA'] = data.Close.rolling(20).mean()
    data['Variance'] = data.Close.rolling(20).std().mean()

    data['Upper_Band'] = data['MA'] + 2 * data['Variance']
    data['Lower_Band'] = data['MA'] - 2 * data['Variance']

    data['trading_signal1'] = np.where(abs(data['Close'] - 2 * data['Variance'])
                                       < data['MA'], 1, -1)

    data['returns1'] = data.Close.pct_change()
    data['Strategy_returns1'] = data.returns1 * data.trading_signal1.shift(1)
    cumulative_returns1 = (data.Strategy_returns1 + 1).cumprod()
    cumulative_returns1.plot(color='g')

    data['%K'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
    data['%D'] = data['%K'].ewm(span=9).mean()

    data['Trading_Signal2'] = np.where(data['%K'] > data['%D'], 1, -1)
    data['returns2'] = data.Close.pct_change()
    data['Strategy_returns2'] = data.returns2 * data.Trading_Signal2.shift(1)
    cumulative_returns2 = (data.Strategy_returns2 + 1).cumprod()
    #  cumulative_returns2.plot(color='r')
    #   plt.show()

    annuals_returns = ((1 + data.Strategy_returns.mean() ** 1 / 3) - 1) * 100
    annuals_returns1 = ((1 + data.Strategy_returns1.mean() ** 1 / 3) - 1) * 100
    annuals_returns2 = ((1 + data.Strategy_returns2.mean() ** 1 / 3) - 1) * 100

    risk_free_rate = 0.06
    daily_risk_free_return = risk_free_rate / 252

    excess_daily_returns = data.returns - daily_risk_free_return

    excess_daily_returns1 = data.returns1 - daily_risk_free_return

    excess_daily_returns2 = data.returns2 - daily_risk_free_return

    sharpe_ratio = (excess_daily_returns.mean() /
                    excess_daily_returns.std()) * np.sqrt(252)

    sharpe_ratio1 = (excess_daily_returns1.mean() /
                     excess_daily_returns1.std()) * np.sqrt(252)

    sharpe_ratio2 = (excess_daily_returns2.mean() /
                     excess_daily_returns2.std()) * np.sqrt(252)

    S = [sharpe_ratio, sharpe_ratio1, sharpe_ratio2, annuals_returns, annuals_returns1, annuals_returns2]
    return S


def affichage(nom_du_fichier):
    x_list = []
    x1_list = []
    x2_list = []
    y_list = []
    y1_list = []
    y2_list = []
    for l in L:
        x_list.append(Sharpe_Ratio(l)[0])
        x1_list.append(Sharpe_Ratio(l)[1])
        x2_list.append(Sharpe_Ratio(l)[2])
        y_list.append(Sharpe_Ratio(l)[3])
        y1_list.append(Sharpe_Ratio(l)[4])
        y2_list.append(Sharpe_Ratio(l)[5])
    plt.clf()
    plt.scatter(x_list, y_list, color='green', marker='+')
    plt.scatter(x1_list, y1_list, color='red', marker='x')
    plt.scatter(x2_list, y2_list, color='yellow', marker='*')
    plt.savefig(nom_du_fichier)


affichage('graphe_sharpe_return.png')










