import pandas as pd
import matplotlib.pyplot as plt
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve

from src.default_modeler import DefaultModeler, undersample
from src.model_mlp import get_weights, define_mlp_model

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
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42, stratify=y_model)

    ## Fit the final gbc model with all training data and the optimized hyperparameters
    print('Fitting the final GBC model')
    ts = time()
    gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=500, random_state=2,
                                    min_samples_leaf=50, max_depth=5, max_features=3)
    dm_gbc = DefaultModeler(gbc)
    dm_gbc.print_model_metrics(X_model, X_holdout, y_model, y_holdout)
    y_pred_gbc = gbc.predict_proba(X_holdout)[:,1]
    fpr_gbc, tpr_gbc, _ = roc_curve(y_holdout, y_pred_gbc)
    score = roc_auc_score(y_holdout, y_pred_gbc)
    print('ROC AUC: %.3f' % score)
    te= time()
    print("Time passed:", te-ts) 

    ## Fit the final Logistic model with all training data
    print('\nFitting the final LG model')
    ts = time()
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_model)
    lg_model = LogisticRegression(solver='lbfgs')
    lg_model.fit(X_std, y_model)
    X_holdout_std = scaler.transform(X_holdout)
    y_pred_lg = lg_model.predict_proba(X_holdout_std)[:,1]
    fpr_lg, tpr_lg, _ = roc_curve(y_holdout, y_pred_lg)
    score = roc_auc_score(y_holdout, y_pred_lg)
    print('ROC AUC: %.3f' % score)
    te= time()
    print("Time passed:", te-ts) 

    ## Fit the final mlp model with all training data
    print('\nFitting the final MLP model')
    ts = time()
    n_input = X_std.shape[1]
    weight_for_0, weight_for_1 = get_weights(y_model)
    weights ={0:weight_for_0, 1:weight_for_1}
    mlp = define_mlp_model(n_input)
    mlp.fit(X_std, y_model, epochs=30, batch_size=2048, verbose=2, class_weight=weights)
    y_pred_mlp = mlp.predict_proba(X_holdout_std)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_holdout, y_pred_mlp)
    score = roc_auc_score(y_holdout, y_pred_mlp)
    print('ROC AUC: %.3f' % score)
    te= time()
    print("Time passed:", te-ts) 

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lg, tpr_lg, label='LR')
    plt.plot(fpr_gbc, tpr_gbc, label='GBC')
    plt.plot(fpr_mlp, tpr_mlp, label='MLP')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('images/roc_curve.png')