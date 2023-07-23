import pandas as pd
import pandas_datareader as web
import datetime as dt


company = 'STM'

# Load Data
start = dt.datetime(2015, 1, 3)
end = dt.datetime(2022, 12, 2)

data = web.DataReader(company, 'yahoo', start, end)

data.to_csv(r'/Users/bastienscheunemann/Desktop/Trading_strategy/Trading model prediction/data.csv')
# Load Test Data
test_start = dt.datetime(2016, 7, 4)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
test_data.to_csv(r'/Users/bastienscheunemann/Desktop/Trading strategy/Trading model prediction/test_data.csv')
