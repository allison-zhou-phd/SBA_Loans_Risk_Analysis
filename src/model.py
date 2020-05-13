import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.externals import joblib

# class DefaultModeler(object):
#     """
#     Default Modeler
#     -------------
#     model: type of sklearn to use, currently set up for Random Forest
#     """

#     def __init__(self, model):
#         self.model = model

def div_count_pos_neg(X, y):
    """Helper function to divide X & y into pos and neg classes and turns counts in each
    
    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D

    Returns
    -------
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
    """Randomly discards negative observations from X & y to achieve the
    target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0.5, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """ 
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    neg_sample_rate = (pos_count*(1 - tp)) / (neg_count * tp)
    neg_keepers = np.random.choice(a=[False, True], size=neg_count, 
                                p=[1-neg_sample_rate, neg_sample_rate])
    X_neg_undersampled = X_neg[neg_keepers]
    y_neg_undersampled = y_neg[neg_keepers]
    X_undersampled = np.vstack((X_neg_undersampled, X_pos))
    y_undersampled = np.concatenate((y_neg_undersampled, y_pos))
    return X_undersampled, y_undersampled

def print_model_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    name = model.__class__.__name__.replace('Classifier','')
    print('*'*30)
    print("{} Accuracy (test):".format(name), accuracy_score(y_test, y_pred))
    print("{} Precision (test):".format(name), precision_score(y_test, y_pred))
    print("{} Recall (test):".format(name), recall_score(y_test, y_pred))

def load_split_data():
    df_loan = pd.read_pickle('data/loan_data')
    df_loan = pd.read_pickle('data/loan_data')
    y = df_loan.pop('Default').values
    X = df_loan.values
    col_names = df_loan.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return (X_train, X_test, y_train, y_test), col_names

if __name__ == "__main__":

    # X_model and y_model to be used in training the model, the holdout sets for model evaluation
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data()

    ### Rely on class_weight option to balance the data
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42, stratify=y_model)

    # # Naive Bayes Model
    # # nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # # print_model_metrics(nb_model, X_train, X_test, y_train, y_test)

    # # Logistic Model
    # lg_model = LogisticRegression(solver='lbfgs', class_weight='balanced')
    # print_model_metrics(lg_model, X_train, X_test, y_train, y_test)

    # # Randome Forest model
    # rfc = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=2, class_weight='balanced')
    # print_model_metrics(rfc, X_train, X_test, y_train, y_test)

    # # Gradient Descend Boost model 
    # gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=300, random_state=2,
    #                                 min_samples_leaf=200, max_depth=3, max_features=3)
    # print_model_metrics(gbc, X_train, X_test, y_train, y_test)
    
    # # AdaBoost Model
    # abc = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', n_estimators=300, random_state=2,
    #                                 min_samples_leaf=200, max_depth=3, max_features=3)
    # print_model_metrics(abc, X_train, X_test, y_train, y_test)

    ### Rely on the undersample method to balance the data
    target_ratio = 0.45
    X_sampled, y_sampled = undersample(X_model, y_model, target_ratio)

