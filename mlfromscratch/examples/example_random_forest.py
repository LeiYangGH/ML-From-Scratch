from __future__ import division, print_function
import numpy as np
from sklearn import datasets
from data_manipulation import train_test_split, normalize, to_categorical,make_diagonal,batch_iterator
from data_operation import accuracy_score,euclidean_distance
from misc import Plot
from random_forest import RandomForest

def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)

    clf = RandomForest(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Random Forest", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()
