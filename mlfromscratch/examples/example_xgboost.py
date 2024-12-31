from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import progressbar
from data_manipulation import train_test_split, normalize, to_categorical,make_diagonal,batch_iterator
from data_operation import accuracy_score,euclidean_distance
from misc import Plot
from xgboost import XGBoost
def main():

    print ("-- XGBoost --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)  

    clf = XGBoost()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, 
        title="XGBoost", 
    accuracy=accuracy, 
    legend_labels=data.target_names)


if __name__ == "__main__":
    main()
