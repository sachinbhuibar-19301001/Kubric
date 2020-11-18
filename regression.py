import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    
    ...
    train_set = pandas.read_csv('linreg_train.csv', header = None).iloc[:, 1:].values.T
    X_train = train_set[:, 0]
    m = numpy.shape(X_train)[0]
    X_train = numpy.matrix([numpy.ones(m), X_train[:]]).T
    Y_train = train_set[:, 1].reshape(m, 1)
    A = numpy.linalg.inv((X_train.T).dot(X_train)).dot(X_train.T).dot(Y_train)
    n = numpy.shape(area)[0]
    area = numpy.matrix([numpy.ones(n), area]).T
    Y_pred = area.dot(A)
    return Y_pred

if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    n = numpy.shape(areas)[0]
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    #rmse = numpy.sqrt(numpy.mean((predicted_prices[:,0] - prices) ** 2))
    total = 0
    for i in range(n):
        total += (predicted_prices[i,0] - prices[i]) ** 2
    rmse = total / n
    rmse = rmse ** 0.5
    print(rmse)
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
