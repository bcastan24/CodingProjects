#A program that uses machine learning to predict the stock market
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import os
import urllib.request, json
import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

dataSource = 'alphavantage'

if dataSource == 'alphavantage':
    apikey = '0VCUU9FT5X9BDFC2'
    ticker = "AAL"
    #getting stock market data from AAL from the last 20 years
    urlString = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,apikey)
    #save the data to a file
    fileToSave = 'stockMarketData-%s.csv'%ticker

    #if I haven't already saved data then store date, low, high, volume, close, open values
    if not os.path.exists(fileToSave):
        with urllib.request.urlopen(urlString) as url:
            data = json.loads(url.read().decode())
            #extract the stock market data that we want
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                dataRow = [date.date(),float(v['3. low']),float(v['2. high']),float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = dataRow
                df.index = df.index + 1
        print("Data saved to: %s"%fileToSave)
        df.to_csv(fileToSave)
    #if the data is already saved then load it from the CSV
    else:
        print("Loading Data from CSV")
        df = pd.read_csv(fileToSave)

#sort data by date
df = df.sort_values('Date')
#double check the result
df.head()

#plotting data
plt.figure(figsize=(18,9))
plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()


