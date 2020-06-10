import numpy as np
import pandas as pd
from time import time
import pickle

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

from src.default_modeler import DefaultModeler, undersample, load_split_data

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    """
    Grid search

    Args:
        estimator: the type of model (e.g. RandomForestClassifier())
        paramter_grid: dictionary defining the gridsearch parameters
        X_train: ndarray - 2D
        y_train: ndarray - 1D
    Returns:  
        Best parameters and model fit with those parameters
    """

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

if __name__ == "__main__":
    
    save_model = 0

    ### Load data with final 5 variables: [Term, U_rate, SBA_g, GrAppv, Sector_Risk]
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data(select=1)
    
    ### Standardize feature variables, add a constant for statsmodel
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_model)
    X_std_sm = sm.add_constant(X_std)

    ### Gridsearch for optimal hyper-parameters
    gridsearch = 0
    if gridsearch:
        X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42, stratify=y_model) 
        gradient_boosting_grid = {'learning_rate': [0.2, 0.1, 0.05],
                                'max_depth': [3, 5],
                                'min_samples_leaf': [50, 100, 200],
                                'max_features': [2, 3],
                                'n_estimators': [300, 500],
                                'random_state': [2]}
        ts = time()
        gbc_best_params, gbc_best_model = gridsearch_with_output(GradientBoostingClassifier(), 
                                                                gradient_boosting_grid, 
                                                                X_train, y_train)
        te= time()
        print("Time passed:", te-ts)
    
    ### Fit final gbc model with all train data and the optimized hyperparameters
    model_gbc = 0
    if model_gbc:
        gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=500, random_state=2,
                                        min_samples_leaf=50, max_depth=5, max_features=3)
        dm_gbc = DefaultModeler(gbc)
        dm_gbc.print_model_metrics(X_model, X_holdout, y_model, y_holdout)
        y_pred = gbc.predict(X_holdout) 
        print(confusion_matrix(y_holdout, y_pred))
        score = roc_auc_score(y_holdout, y_pred)
        print('ROC AUC: %.3f' % score)
        if save_model:
            with open('static/model_gbc.pkl', 'wb') as f: pickle.dump(gbc,f)

    ### Fit final Logistic model with all train data and get the coefficients
    model_lg = 1
    if model_lg:
        lg = LogisticRegression(solver='lbfgs')
        lg.fit(X_std, y_model)
        name = lg.__class__.__name__
        X_holdout_std = scaler.transform(X_holdout)
        y_pred = lg.predict(X_holdout_std)
        print('*'*30)
        print("{} Intercept:".format(name), lg.intercept_) 
        print("{} Coefficients:".format(name), lg.coef_)    
        print("{} Accuracy (test):".format(name), accuracy_score(y_holdout, y_pred))
        print("{} Precision (test):".format(name), precision_score(y_holdout, y_pred))
        print("{} Recall (test):".format(name), recall_score(y_holdout, y_pred))
        if save_model:
            with open('static/model_lg.pkl', 'wb') as f: pickle.dump(lg,f)
    
    ### Fit final Logistic model in statsmodel, get the coefficients and p-value
    stats_model = 1
    if stats_model:
        lg = sm.Logit(y_model, X_std_sm)
        result = lg.fit(method='lbfgs')
        print(result.summary2())

    