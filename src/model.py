import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12})
from time import time

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence
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

def plot_feature_importance(model, X, col_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    name = model.__class__.__name__.replace('Classifier','')
    plt.bar(range(X.shape[1]), importances[indices], color="b")
    plt.title("{} Feature importances".format(name))
    plt.xlabel("Feature")
    plt.ylabel("Feature importance")
    plt.xticks(range(X.shape[1]), col_names[indices], rotation=45, fontsize=12, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array

        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True)
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best

def load_split_data():
    df_loan = pd.read_pickle('data/loan_data')
    feature_choice = ['Term', 'U_rate', 'SBA_g', 'GrAppv', 'SectorRisk', 'Default']
    df_loan = df_loan[feature_choice]
    y = df_loan.pop('Default').values
    X = df_loan.values
    col_names = df_loan.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return (X_train, X_test, y_train, y_test), col_names

if __name__ == "__main__":

    ### X_model and y_model to be used in training the model, the holdout sets for final model evaluation
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data()

    ### Rely on class_weight option to balance the data
    # X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42, stratify=y_model)

    # ### Naive Bayes Model
    # # nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # # print_model_metrics(nb_model, X_train, X_test, y_train, y_test)

    # ### Logistic Model
    # lg_model = LogisticRegression(solver='lbfgs', class_weight='balanced')
    # print_model_metrics(lg_model, X_train, X_test, y_train, y_test)

    # ### Randome Forest model
    # rfc = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=2, class_weight='balanced')
    # print_model_metrics(rfc, X_train, X_test, y_train, y_test)

    # ### Gradient Descend Boost model 
    # gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=300, random_state=2,
    #                                 min_samples_leaf=200, max_depth=3, max_features=3)
    # print_model_metrics(gbc, X_train, X_test, y_train, y_test)
    
    # ### AdaBoost Model
    # abc = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', n_estimators=300, random_state=2,
    #                                 min_samples_leaf=200, max_depth=3, max_features=3)
    # print_model_metrics(abc, X_train, X_test, y_train, y_test)

    ### Rely on the undersample method to balance the data
    target_ratio = 0.45
    X_sampled, y_sampled = undersample(X_model, y_model, target_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled)

    # Logistic Model
    # lg_model = LogisticRegression(solver='lbfgs')
    # print_model_metrics(lg_model, X_train, X_test, y_train, y_test)

    ### Randome Forest model - feature importance
    # rfc = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=2,)
    # print_model_metrics(rfc, X_train, X_test, y_train, y_test)
    # names = ['StateRisk', 'SectorRisk','Term', 'NumEmp', 'LowDocu', 'GrAppv', 'SBA_g','U_rate']
    """Below code would hang when running. No error msg.  Need to investigate
    # features = [2, 5, 7, 3, 6]
    # plot_partial_dependence(rfc, X_train, features=features, feature_names=names) 
    # fig = plt.gcf()
    # fig.set_size_inches(11,8)
    # plt.tight_layout()
    # plt.savefig('images/rf_partDepend.png')
    # plt.close()
    """
    # plot_feature_importance(rfc, X_sampled, col_names)
    # plt.savefig('images/rfc_feature_importances.png')
    # plt.close()


    ### Gradient Boost model - feature importance
    # gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=300, random_state=2,
    #                                 min_samples_leaf=200, max_depth=3, max_features=3)
    # print_model_metrics(gbc, X_train, X_test, y_train, y_test)
    # features = [2, 7, 5, 6, 1]
    # plot_partial_dependence(gbc, X_train, features=features, feature_names=names) 
    # fig = plt.gcf()
    # fig.set_size_inches(11,8)
    # plt.tight_layout()
    # plt.savefig('images/gbc_partDepend.png')
    # plt.close()
    # plot_feature_importance(gbc, X_sampled, col_names)
    # plt.savefig('images/gbc_feature_importances.png')
    # plt.close()

    # AdaBoost Model
    # abc = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', n_estimators=300, random_state=2,
    #                                 min_samples_leaf=200, max_depth=3, max_features=3)
    # print_model_metrics(abc, X_train, X_test, y_train, y_test)

    ## Reduce model to 5 variables: Term, U_rate, SBA_g, GrAppv, Sector_Risk
    ## Conduct gridSearch to find the best fitting gbc model
    # gradient_boosting_grid = {'learning_rate': [0.2, 0.1, 0.05],
    #                           'max_depth': [3, 5],
    #                           'min_samples_leaf': [50, 200],
    #                           'max_features': [2, 3],
    #                           'n_estimators': [300, 500],
    #                           'random_state': [2]}
    # ts = time()
    # gdbr_best_params, gdbr_best_model = gridsearch_with_output(GradientBoostingClassifier(), 
    #                                                            gradient_boosting_grid, 
    #                                                            X_train, y_train)
    # te= time()
    # print("Time passed:", te-ts)
    
    ## Fit final gbc model with all train data and the optimized hyperparameters
    # gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=500, random_state=2,
    #                                 min_samples_leaf=50, max_depth=5, max_features=3)
    # print_model_metrics(gbc, X_sampled, X_holdout, y_sampled, y_holdout)

    # Fit final Logistic model with all train data and get the coefficients
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_sampled)
    lg_model = LogisticRegression(solver='saga')
    lg_model.fit(X_std, y_sampled)
    print("Coefficients:", lg_model.coef_)
    X_holdout_std = scaler.transform(X_holdout)
    y_pred = lg_model.predict(X_holdout_std)
    name = lg_model.__class__.__name__
    print('*'*30)
    print("{} Accuracy (test):".format(name), accuracy_score(y_holdout, y_pred))
    print("{} Precision (test):".format(name), precision_score(y_holdout, y_pred))
    print("{} Recall (test):".format(name), recall_score(y_holdout, y_pred))

    