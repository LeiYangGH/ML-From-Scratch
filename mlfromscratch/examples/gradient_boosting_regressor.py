from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar

from data_manipulation import train_test_split, normalize, to_categorical,make_diagonal,batch_iterator,divide_on_feature,get_random_subsets,to_categorical
from data_operation import accuracy_score,euclidean_distance,calculate_entropy,mean_squared_error
from misc import Plot
from loss_functions import SquareLoss
from misc import bar_widgets
from gradient_boosting import GradientBoostingClassifier,GradientBoostingRegressor


def main():
    print ("-- Gradient Boosting Regression --")

    # Load temperature data
    data = pd.read_csv('../data/TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = np.atleast_2d(data["temp"].values).T

    X = time.reshape((-1, 1))               # Time. Fraction of the year [0, 1]
    X = np.insert(X, 0, values=1, axis=1)   # Insert bias term
    y = temp[:, 0]                          # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()