from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pandas_datareader.data import DataReader
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/validate', methods = ['POST','GET'])
def validate():
    if request.method == 'POST':
        if request.form.get('username') == 'abc' and request.form.get('password') == '123':
            return redirect(url_for('index'))

@app.route('/index')
def index():
    stock_list = ['Apple','Amazon','Google','Microsoft','NVIDIA','Tesla']
    return render_template('index.html', data = stock_list)

@app.route('/predict', methods = ['POST','GET'])
def predict():
    global df, symbol
    if request.method == 'POST':
        stock_dict = {'Apple':'AAPL','Amazon':'AMZN','Google':'GOOG','Microsoft':'MSFT','NVIDIA':'NVDA','Tesla':'TSLA'}
        c_name = request.form.get('c_name')
        symbol = stock_dict[c_name]
        end= datetime.now()
        start = datetime(end.year - 1, end.month, end.day)
        df = pdr.get_data_yahoo("AMZN", start=start, end=end)
        print(df)
        #df = web.DataReader(symbol,data_source='yahoo',start=start,end=end)
        return redirect(url_for('open'))

@app.route('/open')
def open():
    plt.figure(figsize=(16,8))
    plt.title('Open Price History')
    plt.plot(df['Open'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Open Price USD($)',fontsize=18)
    plt.savefig('static/open.jpg')
    return render_template('result.html', path = 'static/open.jpg')

@app.route('/close')
def close():
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD($)',fontsize=18)
    plt.savefig('static/close.jpg')
    return render_template('result.html', path = 'static/close.jpg')

@app.route('/high')
def high():
    plt.figure(figsize=(16,8))
    plt.title('High Price History')
    plt.plot(df['High'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('High Price USD($)',fontsize=18)
    plt.savefig('static/high.jpg')
    return render_template('result.html', path = 'static/high.jpg')

@app.route('/low')
def low():
    plt.figure(figsize=(16,8))
    plt.title('Low Price History')
    plt.plot(df['Low'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Low Price USD($)',fontsize=18)
    plt.savefig('static/low.jpg')
    return render_template('result.html', path = 'static/low.jpg')

@app.route('/volume')
def volume():
    plt.figure(figsize=(16,8))
    plt.title('Volume')
    plt.plot(df['Volume'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Sales Volume',fontsize=18)
    plt.savefig('static/volume.jpg')
    return render_template('result.html', path = 'static/volume.jpg')

@app.route('/future')
def future():
    end= datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    df = pdr.get_data_yahoo("AMZN", start=start, end=end)
    #df = web.DataReader(symbol,data_source='yahoo',start=start,end=end)
    print(df.head())

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    print(training_data_len)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    print(scaled_data)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(rmse)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig('static/future.jpg')
    return render_template('result.html', path = 'static/future.jpg')

if __name__ == '__main__':
    app.run(debug=True,port=2001)