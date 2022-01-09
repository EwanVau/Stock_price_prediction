# I, Ewan Vaughan, have read and understood the School's Academic Integrity Policy,
# as well as guidance relating to this module, and confirm that this submission
# complies with the policy. The content of this file is my own original work,
# with any significant material copied or adapted from other sources clearly
# indicated and attributed.

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# add_datepart needs to be deleted if not used.
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    """Currently seems to be not working due to being a depricated function.

    This function is part of the fastai package, I was unable to import along with
    other packages and as such I have pulled this function from the fastai github,
    found at https://github.com/fastai/fastai .

    add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string or list of strings that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    >>>df2 = pd.DataFrame({'start_date' : pd.to_datetime(['3/11/2000','3/13/2000','3/15/2000']),
                            'end_date':pd.to_datetime(['3/17/2000','3/18/2000','4/1/2000'],infer_datetime_format=True)})
    >>>df2
        start_date	end_date
    0	2000-03-11	2000-03-17
    1	2000-03-13	2000-03-18
    2	2000-03-15	2000-04-01
    >>>add_datepart(df2,['start_date','end_date'])
    >>>df2
    	start_Year	start_Month	start_Week	start_Day	start_Dayofweek	start_Dayofyear	start_Is_month_end	start_Is_month_start	start_Is_quarter_end	start_Is_quarter_start	start_Is_year_end	start_Is_year_start	start_Elapsed	end_Year	end_Month	end_Week	end_Day	end_Dayofweek	end_Dayofyear	end_Is_month_end	end_Is_month_start	end_Is_quarter_end	end_Is_quarter_start	end_Is_year_end	end_Is_year_start	end_Elapsed
    0	2000	    3	        10	        11	        5	            71	            False	            False	                False	                False	                False	            False	            952732800	    2000	    3	        11	        17	    4	            77	            False	            False	            False	            False	                False	        False	            953251200
    1	2000	    3	        11	        13	        0	            73	            False	            False	                False	                False               	False           	False           	952905600     	2000       	3	        11      	18  	5           	78          	False	            False           	False           	False               	False          	False           	953337600
    2	2000	    3	        11	        15	        2           	75          	False           	False               	False               	False               	False               False           	953078400      	2000    	4          	13      	1   	5           	92          	False           	True            	False           	True                	False          	False           	954547200
    """
    if isinstance(fldnames, str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)


def RMSE(prediction, target):
    """"Basic function which returns the Root Mean Squared Error of a predicted
    value and the true value."""
    return np.sqrt(np.mean(np.power((prediction - target), 2)))


def predicterror(predict):
    """This function simply calculates the difference between the predicted value
    and the true value."""
    return (abs(predict["Close"] - predict["Predictions"]))


def datefix(dataset, lstm=False):
    """This function generates a pandas dataframe with the relevant columns used in
    this analysis. This also formats the date as d/m/Y and moves it to the index
    instead of existing as its own column."""
    if lstm == True:
        dataset.index = dataset["Date"]
        dataset.drop("Date", axis=1, inplace=True)
    else:
        dataset = dataset[["Date", "Open", "High", "Low", "Last", "Close", "Total Trade Quantity"]]
        dataset["Date"] = pd.to_datetime(dataset["Date"], format="%d/%m/%Y")
        dataset.index = dataset["Date"]
        dataset = dataset.sort_index(ascending=True, axis=0)
        dataset.drop("Date", axis=1, inplace=True)
    return (dataset)


def handle_data(dataset, data_out=True, lstm=False, test_percent=0.2):
    """This modifies the raw dataset into one which will be used throughout this
    project and also the relevant testing and validation datasets.

    dataset: the dataset which you wish to modify and extract test and
    validation datasets from.

    test_percent: the percentage of testing data you want to allocate. 20% is
    allocated by default. Input must be between 0 and 1.

    lstm: only used in the 'Predict_LSTM' function, False by default."""
    inds = int((1 - test_percent) * len(dataset))
    if lstm == True:
        dataout = datefix(dataset, lstm=True)
        train = dataset.iloc[:inds, :]
        test = dataset.iloc[inds:, :]
    else:
        dataout = datefix(dataset)
        train = dataout.iloc[:inds, :]
        test = dataout.iloc[inds:, :]
    if data_out == True:
        return (dataout, train, test)
    else:
        return (train, test)


def handle_predict(train, test):
    """This function splits the test and training data into datasets that are
    use in the various models for prediction. This splits the close column into
    the y dataset."""
    x_tr, x_te = [], []
    x_tr = train.drop("Close", axis=1)
    y_tr = pd.DataFrame(train["Close"], columns=["Close"])
    x_te = test.drop("Close", axis=1)
    y_te = pd.DataFrame(test["Close"], columns=["Close"])
    return (x_tr, y_tr, x_te, y_te)


# Function not currently used.
def forecast(start_y, start_m, start_d, n):
    """This functions generates dates used in forcasting predictions. This function
    will only generate days that are weekdays to simulate real trading days. This
    does not account for holidays.

    start_y: year of starting date
    start_m: month of starting date
    start_d: day of starting date
    n: number of days from starting date
    """
    date_start = "{}/{}/{}".format(start_d, start_m, start_y)
    date_start = pd.to_datetime(date_start, format="%d/%m/%Y")
    datelist = pd.DataFrame(columns=["Date"])
    dateout = []
    for i in range(0, n):
        new_date = date_start + datetime.timedelta(days=i)
        weekday = new_date.weekday()
        if weekday >= 5:
            continue
        datelist = np.append(datelist, new_date)
    dateout = pd.DataFrame(datelist)
    dateout.rename_axis("Date", columns="0")
    return dateout


# Plotting functions
def volume_plot(dataset):
    """Plots the initial dataset along with the volume in a subplot"""
    dataset, train, test = handle_data(dataset, data_out=True)
    x = dataset.index
    xval = test.index[0]
    plt.rcParams["figure.figsize"] = 25, 10
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    fig.suptitle("Plots of NSE-TATAGLOBAL Closing Price and Volume of Stocks Traded vs Date")
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    ax1.plot(x, dataset["Close"], "b-", label="Data set")
    ax1.plot(x, dataset["High"], "g-", alpha=0.4, label="High")
    ax1.plot(x, dataset["Low"], "g-", alpha=0.4, label="Low")
    ax1.axvline(xval, color="black", alpha=0.4, linestyle="--", label="Training and Testing set divider")
    ax1.fill_between(x, dataset["High"], dataset["Low"], alpha=0.3, color="green")
    ax1.set_ylabel("Stock Price (USD)/$")
    ax1.legend(loc=0, fontsize=12)

    ax2.bar(x, dataset["Total Trade Quantity"], align="center", width=0.5, color="black", label="Traded Stocks")
    ax2.axvline(xval, color="black", alpha=0.4, linestyle="--", label="Training and Testing set divider")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Sctocks Traded per Day")
    ax2.legend(loc=0, fontsize=12)
    fig.show()
    return


def multi_predict_plot(dataset, lr, knn, lstm):
    """Plots all of the predictions into a sub plot which includes the entire
    dataset and a subplot for just the testing data."""
    dataset, train, test = handle_data(dataset)
    x1 = train.index
    x2 = test.index

    plt.rcParams["figure.figsize"] = 25, 10
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    fig.suptitle("Plots of Different Predictions vs Target Price")
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(x1, train["Close"], "k-", label="Closing Price")
    ax1.plot(x2, lr["Close"], "b-", label="Target Price")
    ax1.plot(x2, lr["Predictions"], "g-", label="Linear Regression")
    ax1.plot(x2, knn["Predictions"], "r-", label="KNN")
    ax1.plot(x2, lstm["Predictions"], "m-", label="LSTM")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price (USD)/$")
    ax1.legend(loc=0, fontsize=12)

    ax2.plot(x2, lr["Close"], "b-", label="Target Prices")
    ax2.plot(x2, lr["Predictions"], "g-", label="Linear Regression")
    ax2.plot(x2, knn["Predictions"], "r-", label="KNN")
    ax2.plot(x2, lstm["Predictions"], "m-", label="LSTM")
    ax2.set_ylabel("Stock Price (USD)/$")
    ax2.legend(loc=0, fontsize=12)
    fig.show()
    return


def errorplot(dataset, lr, knn, lstm):
    """Plots the difference between the predicted value and the true value for
    each model. Also includes a cumulative error plot."""
    train, test = handle_data(dataset, data_out=False)
    x2 = test.index
    lr_err = predicterror(lr)
    lr_rmse = RMSE(lr["Predictions"], lr["Close"])
    lr_cumerr = np.cumsum(lr_err)
    knn_err = predicterror(knn)
    knn_rmse = RMSE(knn["Predictions"], knn["Close"])
    knn_cumerr = np.cumsum(knn_err)
    lstm_err = predicterror(lstm)
    lstm_rmse = RMSE(lstm["Predictions"], lstm["Close"])
    lstm_cumerr = np.cumsum(lstm_err)

    plt.rcParams["figure.figsize"] = 25, 10
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    fig.suptitle("Plot of Absolute Error and Cumulative Error")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(x2, lr_err, "g-", label="Linear Regression")
    ax1.axhline(lr_rmse, color="g", ls="--", alpha=0.5, label="Linear Regression RMSE")
    ax1.plot(x2, knn_err, "r-", label="KNN")
    ax1.axhline(knn_rmse, color="r", ls="--", alpha=0.5, label="KNN RMSE")
    ax1.plot(x2, lstm_err, "m-", label="LSTM")
    ax1.axhline(lstm_rmse, color="m", ls="--", alpha=0.5, label="LSTM RMSE")
    ax1.title.set_text("Difference in Predicted Values and the True Values")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Absolute Error")
    ax1.legend(loc=0, fontsize=12)

    ax2.plot(x2, lr_cumerr, "g-", label="Linear Regression")
    ax2.plot(x2, knn_cumerr, "r-", label="KNN")
    ax2.plot(x2, lstm_cumerr, "m-", label="LSTM")
    ax2.title.set_text("Cumulative Error Plot")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Error")
    ax2.legend(loc=0, fontsize=12)
    fig.show()
    return


# Prediction Functions
def Predict_LR(dataset, extra_dates=False):
    """This function predicts test data by using the training data to train a
    linear regression model.

    dataset: the dataset for which you want to predict prices for
    extra_dates: adds extra features related to the date eg. 'is_month_start',
                 'is_year_end' etc.
    """
    start = time.time()
    train, test = handle_data(dataset, data_out=False, test_percent=0.2)
    if extra_dates == True:
        add_datepart(train, "Index")
        train.drop("Elapsed", axis=1, inplace=True)
        add_datepart(test, "Index")
        test.drop("Elapsed", axis=1, inplace=True)

    x_train, y_train, x_test, y_test = handle_predict(train, test)
    # x_train,  x_test, y_train, y_test = train_test_split(Data.drop("Close", axis=1), Data["Close"], test_size=0.2)

    model = LinearRegression()
    model.fit(x_train, y_train)
    lr_predict = model.predict(x_test)

    lr_RMSE = RMSE(lr_predict, y_test)
    lr_result = test
    lr_result["Predictions"] = 0
    lr_result["Predictions"] = lr_predict

    end = time.time()
    print("Process lasted {} seconds.".format(end - start))
    return (lr_result, lr_RMSE)


def Predict_KNN(dataset):
    """This function predicts test data by using the training data to train a
    K-Nearest Neighbour regression model.

    dataset: the dataset for which you want to predict prices for
    """
    start = time.time()
    scaler = MinMaxScaler(feature_range=(0, 1))

    train, test = handle_data(dataset, data_out=False, test_percent=0.2)
    x_train, y_train, x_test, y_test = handle_predict(train, test)

    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.fit_transform(x_test))
    y_train = pd.DataFrame(scaler.fit_transform(y_train))

    model = neighbors.KNeighborsRegressor()

    model.fit(x_train, y_train)
    KNN_predict = model.predict(x_test)
    KNN_predict = scaler.inverse_transform(KNN_predict)
    KNN_RMSE = RMSE(KNN_predict, y_test)

    KNN_result = test
    KNN_result["Predictions"] = 0
    KNN_result["Predictions"] = KNN_predict

    end = time.time()
    print("Process lasted {} seconds.".format(end - start))
    return (KNN_result, KNN_RMSE)


def Predict_LSTM(dataset, past_data=60):
    """This function predicts test data by using the training data to train a
    Long Short Term Memory regression model.

    dataset: the dataset of the stock you wish to predict for
    past_data: the amount of past data values to appear in the testing dataset
    """
    start = time.time()
    scaler = MinMaxScaler(feature_range=(0, 1))

    lstm_data = pd.DataFrame(index=range(0, len(dataset)), columns=['Date', 'Close'])
    for i in range(0, len(dataset)):
        lstm_data["Date"][i] = dataset["Date"][i]
        lstm_data["Close"][i] = dataset["Close"][i]

    modified_dataset, train, test = handle_data(lstm_data, data_out=True, lstm=True, test_percent=0.2)

    # Scale the data down due to the sensitivity of the activation function
    scaled_data = scaler.fit_transform(lstm_data)

    x_train, y_train = [], []
    for i in range(past_data, len(train)):
        x_train.append(scaled_data[i - past_data:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train, ndmin=1), np.array(y_train, ndmin=1)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Creating the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)

    # predicition values, using past data from the training data
    inputs = lstm_data[len(dataset) - len(test) - past_data:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.fit_transform(inputs)

    X_test = []
    for i in range(past_data, inputs.shape[0]):
        X_test.append(inputs[i - past_data:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    lstm_predict = model.predict(X_test)
    lstm_predict = scaler.inverse_transform(lstm_predict)

    lstm_RMSE = RMSE(test, lstm_predict)
    train, test = handle_data(dataset, data_out=False, test_percent=0.2)
    lstm_result = test
    lstm_result["Predictions"] = 0
    lstm_result["Predictions"] = lstm_predict

    end = time.time()
    print("Process lasted {} seconds".format(end - start))
    return (lstm_result, lstm_RMSE)


############################################
# This reads from the folder "Datasets"
Tata = pd.read_csv(r".\Datasets\NSE-TATAGLOBAL11.csv")

# Plotting the dataset along with the volume of stocks traded on that date
volume_plot(Tata)

# Prediction using Linear Regression
LR_Result, LR_Error = Predict_LR(Tata)

# Prediction using K-Nearest Neighbours Regressor
KNN_Result, KNN_error = Predict_KNN(Tata)

# Prediction using Long Short Term Memory
LSTM_Result, LSTM_error = Predict_LSTM(Tata)

# Plotting all the predictions into a subplot
multi_predict_plot(Tata, LR_Result, KNN_Result, LSTM_Result)

# Plotting the error in predictions and the cumulative error plot
errorplot(Tata, LR_Result, KNN_Result, LSTM_Result)