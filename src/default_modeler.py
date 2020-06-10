import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class DefaultModeler(object):
    """
    Default Modeler, sklearn model to use for modeling a binary default risk

    """

    def __init__(self, model):
        self.model = model

    def print_model_metrics(self, X_train, X_test, y_train, y_test):
        """
        Print model performance metrics

        Args:
            X_train: ndarray - 2D
            X_test: ndarray - 2D
            y_train: ndarray - 1D
            y_test: ndarray - 1D
        Returns:
            Nothing, just prints
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        name = self.model.__class__.__name__.replace('Classifier','')
        print('*'*30)
        print("{} Accuracy (test):".format(name), accuracy_score(y_test, y_pred))
        print("{} Precision (test):".format(name), precision_score(y_test, y_pred))
        print("{} Recall (test):".format(name), recall_score(y_test, y_pred))

    def plot_feature_importance(self, X, col_names):
        """
        Plots feature importance (for random forest and gradient boost models)

        Args:
            X: ndarray - 2D
            col_names(list): column names of X
        Returns:
            Feature importance plot
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        name = self.model.__class__.__name__.replace('Classifier','')
        plt.bar(range(X.shape[1]), importances[indices], color="b")
        plt.title("{} Feature Importances".format(name))
        plt.xlabel("Feature")
        plt.ylabel("Feature importance")
        plt.xticks(range(X.shape[1]), col_names[indices], rotation=45, fontsize=12, ha='right')
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()

def div_count_pos_neg(X, y):
    """
    Helper function to divide X & y into pos and neg classes and turns counts in each
    
    Args:  
        X : ndarray - 2D
        y : ndarray - 1D
    Returns:
        negative_count : Int
        positive_count : Int
        X_positives    : ndarray - 2D
        X_negatives    : ndarray - 2D
        y_positives    : ndarray - 1D
        y_negatives    : ndarray - 1D
    """

    neg, pos = y==0, y==1
    neg_count, pos_count = np.sum(neg), np.sum(pos)
    X_pos, y_pos = X[pos], y[pos]
    X_neg, y_neg = X[neg], y[neg]
    return neg_count, pos_count, X_pos, X_neg, y_pos, y_neg

def undersample(X, y, tp):
    """
    Randomly discards negative observations from X & y to achieve the
    target proportion of positive to negative observations.

    Args:
        X  : ndarray - 2D
        y  : ndarray - 1D
        tp : float - range [0.5, 1], target proportion of positive class observations
    Returns:
        X_undersampled : ndarray - 2D
        y_undersampled : ndarray - 1D
    """ 
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    neg_sample_rate = (pos_count*(1 - tp)) / (neg_count * tp)
    np.random.seed(3)
    neg_keepers = np.random.choice(a=[False, True], size=neg_count, 
                                p=[1-neg_sample_rate, neg_sample_rate])
    X_neg_undersampled = X_neg[neg_keepers]
    y_neg_undersampled = y_neg[neg_keepers]
    X_undersampled = np.vstack((X_neg_undersampled, X_pos))
    y_undersampled = np.concatenate((y_neg_undersampled, y_pos))
    return X_undersampled, y_undersampled

def load_split_data(select=0):
    """ 
        Load data in
    Args:
        select(int): option to control whether selective features will be used 
    Returns:
        Train_test datasets for X and y, as well as a list for column names
    """
    df_loan = pd.read_pickle('data/loan_data')
    if select:
        feature_choice = ['Term', 'GrAppv', 'U_rate', 'SBA_g', 'SectorRisk', 'Default']
        df_loan = df_loan[feature_choice]

    y = df_loan.pop('Default').values
    X = df_loan.values
    col_names = df_loan.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return (X_train, X_test, y_train, y_test), col_names