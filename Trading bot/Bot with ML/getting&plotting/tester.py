import pandas as pd
import plotly as py
import plotly.graph_objs as go
from plotly import subplots
from Experimentation.functions import *


# (1) Load up our data & create moving average

df = pd.read_csv('DATA/EURUSD_Candlestick_1_D_ASK_03.02.2020-31.03.2020.csv')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df.date = pd.to_datetime(df.date, format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open', 'high', 'low', 'close', 'volume']]
#df['Symbol'] = 'EURUSD'
df = df.drop_duplicates(keep=False)
df = df.iloc[1:]

ma = df.close.rolling(center=False, window=30).mean()

# (2) Get function data from selected function:

f = Fourier(df, [10, 15], method='difference')
res = f.results

# (3) Plot
trace0 = go.Ohlc(x=df.index.to_pydatetime(), open=df.open, high=df.high, low=df.low, close=df.close, name='Currency Quote')
trace1 = go.Scatter(x=df.index.to_pydatetime(), y=ma)
trace2 = go.Scatter(x=res.index, y=res.high)

data = [trace0, trace1, trace2]

fig = py.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)


py.offline.plot(fig, filename='trace.html')

