from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from linear_discriminant_analysis import LDA
from data_manipulation import train_test_split, normalize, to_categorical,make_diagonal,batch_iterator
from data_operation import accuracy_score,euclidean_distance
from misc import Plot

def main():
    # Load the dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Three -> two classes
    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Fit and predict using LDA
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="LDA", accuracy=accuracy)

if __name__ == "__main__":
    main()
