from __future__ import division, print_function
from sklearn import datasets
import numpy as np
from data_manipulation import train_test_split, normalize, to_categorical,make_diagonal
from data_operation import accuracy_score
from misc import Plot
from naive_bayes import NaiveBayes

def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Naive Bayes", accuracy=accuracy, legend_labels=data.target_names)

if __name__ == "__main__":
    main()