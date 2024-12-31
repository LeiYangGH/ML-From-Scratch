from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Import helper functions
from data_manipulation import train_test_split, normalize, to_categorical,make_diagonal,batch_iterator,divide_on_feature,get_random_subsets,to_categorical
from data_operation import accuracy_score,euclidean_distance,calculate_entropy,calculate_covariance_matrix
from misc import Plot
from gradient_boosting import GradientBoostingClassifier

def main():

    print ("-- Gradient Boosting Classification --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)


    Plot().plot_in_2d(X_test, y_pred, 
        title="Gradient Boosting", 
        accuracy=accuracy, 
        legend_labels=data.target_names)



if __name__ == "__main__":
    main()