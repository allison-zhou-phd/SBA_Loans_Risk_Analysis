import pandas as pd
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.default_modeler import DefaultModeler, undersample

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

def load_split_data():
    """ 
        Load data in 
    Args:
        train(int): option to control whether all features will be used
    Returns:
        Train_test datasets for X and y, as well as a list for column names
    """
    df_loan = pd.read_pickle('data/loan_data')
    feature_choice = ['Term', 'GrAppv', 'U_rate', 'SBA_g', 'SectorRisk', 'Default']
    df_loan = df_loan[feature_choice]

    y = df_loan.pop('Default').values
    X = df_loan.values
    col_names = df_loan.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return (X_train, X_test, y_train, y_test), col_names

if __name__ == "__main__":
    
    ### Reduce model to 5 variables: [Term, U_rate, SBA_g, GrAppv, Sector_Risk], conduct gridSearch to find the best fitting gbc model
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data()
    target_ratio = 0.45
    X_sampled, y_sampled = undersample(X_model, y_model, target_ratio)
    # X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled)

    random_forest_grid = {'learning_rate': [0.3, 0.2, 0.1],
                          'max_depth': [3, 5],
                          'min_samples_leaf': [50, 100, 200],
                          'max_features': [2, 3],
                          'n_estimators': [300, 500],
                          'random_state': [2]}
    ts = time()
    rf_best_params, rf_best_model = gridsearch_with_output(RandomForestClassifier(), 
                                                               random_forest_grid, 
                                                               X_sampled, y_sampled)
    te= time()
    print("Time passed:", te-ts)
    
    ## Fit final gbc model with all train data and the optimized hyperparameters
    # gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=500, random_state=2,
    #                                 min_samples_leaf=50, max_depth=5, max_features=3)
    # dm_gbc = DefaultModeler(gbc)
    # dm_gbc.print_model_metrics(X_sampled, X_holdout, y_sampled, y_holdout)

    # ## Fit final Logistic model with all train data and get the coefficients
    # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # X_std = scaler.fit_transform(X_sampled)
    # lg_model = LogisticRegression(solver='lbfgs')
    # lg_model.fit(X_std, y_sampled)
    # name = lg_model.__class__.__name__
    # X_holdout_std = scaler.transform(X_holdout)
    # y_pred = lg_model.predict(X_holdout_std)
    # print('*'*30)
    # print("{} Coefficients:".format(name), lg_model.coef_)    
    # print("{} Accuracy (test):".format(name), accuracy_score(y_holdout, y_pred))
    # print("{} Precision (test):".format(name), precision_score(y_holdout, y_pred))
    # print("{} Recall (test):".format(name), recall_score(y_holdout, y_pred))