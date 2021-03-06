import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import pickle
import matplotlib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model

from src.default_modeler import DefaultModeler, undersample, load_split_data
#from src.model_mlp import get_weights, define_mlp_model

def plot_model_profits(model_profits, save_path=None):
    """
        Plotting function to compare profit curves of different models.
    Args:
        model_profits : list((model, profits, thresholds))
        save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    Returns:
        None
    """
    _, ax = plt.subplots()
    for model_str, _, profits, _ in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        ax.plot(percentages, profits, label=model_str)

    ax.set_title("Profit Curve")
    ax.set_xlabel("Percentage of test instances (decreasing by score)", fontsize=12)
    ax.set_ylabel("Profit ($)", fontsize=12)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def find_best_threshold(model_profits):
    """
        Find model-threshold combo that yields highest profit.
    Args:
        model_profits : list((model, profits, thresholds))
    Returns:
        max_model(str): the model that maximizes profits
        max_threshold(float) : threshold that maximizes profits
        max_profit(float)    : maximized profits
    """
    max_model = None
    max_threshold = None
    max_profit = None
    for _, model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit

if __name__ == "__main__":
    
    cost_benefit = np.array([[50000, -5000], [0, 0]])
    model_profits = []

    ### Load data with final 5 variables: [Term, U_rate, SBA_g, GrAppv, Sector_Risk]
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data(select=1)

    ### Standarsize feature variables (needed for logistic regression and MLP) 
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_model)
    X_holdout_std = scaler.transform(X_holdout)
   
    ### Load the final gbc model
    model_gbc = 1
    if model_gbc:
        print('Load GBC model')
        ts = time()
        with open('static/model_gbc.pkl', 'rb') as f_gbc:
            gbc = pickle.load(f_gbc)
        y_pred_gbc = gbc.predict_proba(X_holdout)[:,1]
        fpr_gbc, tpr_gbc, thresholds_gbc = roc_curve(y_holdout, y_pred_gbc)
        profit_gbc = tpr_gbc * cost_benefit[0,0] + fpr_gbc * cost_benefit[0,1]
        model_profits.append(('GradientBoostingClassifier', gbc, profit_gbc, thresholds_gbc))
        score = roc_auc_score(y_holdout, y_pred_gbc)
        print('ROC AUC: %.3f' % score)
        te= time()
        print("Time passed:", te-ts) 

    ### Load the final Logistic model
    model_lg = 0
    if model_lg:
        print('\nLoad LR model')
        ts = time()
        with open('static/model_lg.pkl', 'rb') as f_lg:
            lg = pickle.load(f_lg)
        y_pred_lg = lg.predict_proba(X_holdout_std)[:,1]
        fpr_lg, tpr_lg, thresholds_lg = roc_curve(y_holdout, y_pred_lg)
        profit_lg = tpr_lg * cost_benefit[0,0] + fpr_lg * cost_benefit[0,1]
        model_profits.append(('LogisticRegression', lg, profit_lg, thresholds_lg))
        score = roc_auc_score(y_holdout, y_pred_lg)
        print('ROC AUC: %.3f' % score)
        te= time()
        print("Time passed:", te-ts) 

    ### Load the final mlp model
    model_mlp = 0
    if model_mlp:
        print('\nLoad MLP model')
        ts = time()
        mlp = load_model("static/model_mlp.h5")
        y_pred_mlp = mlp.predict_proba(X_holdout_std)
        fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_holdout, y_pred_mlp)
        profit_mlp = tpr_mlp * cost_benefit[0,0] + fpr_mlp * cost_benefit[0,1]
        model_profits.append(('MultiLayerPerceptron', mlp, profit_mlp, thresholds_mlp))
        score = roc_auc_score(y_holdout, y_pred_mlp)
        print('ROC AUC: %.3f' % score)
        te= time()
        print("Time passed:", te-ts) 

    ### Plot the ROC Curve
    plot_roc = 0
    if plot_roc:
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_gbc, tpr_gbc, label='GradientBoostingClassifier')
        plt.plot(fpr_lg, tpr_lg, label='LogisticRegression')
        plt.plot(fpr_mlp, tpr_mlp, label='MultiLayerPerceptron')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.savefig('images/roc_curve.png')
        plt.close()

    ### Plot the Profit Curve
    plot_profit = 1
    if plot_profit:
        if model_lg and model_mlp:
            plot_model_profits(model_profits, 'images/profit_curve.png')
        else:
            plot_model_profits(model_profits, 'images/profit_curve_1model.png')

    ### Find the max profit and corresponding model and threshold
    find_max = 0
    if find_max:
        print('\nFinding the max profits...')
        ts = time()
        max_model, max_thresh, max_profit = find_best_threshold(model_profits)
        if max_model == gbc:
            max_labeled_positives = max_model.predict_proba(X_holdout) >= max_thresh
        else:
            max_labeled_positives = max_model.predict_proba(X_holdout_std) >= max_thresh
        proportion_positives = max_labeled_positives.mean()
        reporting_string = ('Best model:\t\t{}\n'
                            'Best threshold:\t\t{:.2f}\n'
                            'Resulting profit:\t{}\n'
                            'Proportion positives:\t{:.2f}')
        print(reporting_string.format(max_model.__class__.__name__, max_thresh,
                                    max_profit, proportion_positives))
        te= time()
        print("Time passed:", te-ts) 