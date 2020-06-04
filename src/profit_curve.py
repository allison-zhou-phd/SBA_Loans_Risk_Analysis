import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.model_mlp import get_weights, define_mlp_model

def standard_confusion_matrix(y_true, y_pred):
    """
        Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Args:
        y_true : ndarray - 1D
        y_pred : ndarray - 1D
    Returns:
        confusion_matrix: ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit, predicted_probs, targets):
    """
        Function to calculate list of profits based on supplied cost-benefit
        matrix and prediced probabilities of data points and thier true labels.
    Args:
        cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
        predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
        targets          : ndarray - 1D, true label of datapoints, 0 or 1
    Returns:
        profits    : ndarray - 1D
        thresholds : ndarray - 1D
    """
    n_obs = float(len(targets))
    # Make sure that 1 is going to be one of our thresholds
    maybe_one = [] if 1 in predicted_probs else [1] 
    thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(targets, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)

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
    for model, profits, _ in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=model.__class__.__name__)

    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
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
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit

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
    cost_benefit = np.array([[160, -55], [0, 0]])
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data()
    
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_model)
    X_holdout_std = scaler.transform(X_holdout)
    
    n_input = X_std.shape[1]
    weight_for_0, weight_for_1 = get_weights(y_model)
    weights ={0:weight_for_0, 1:weight_for_1}

    # Define and fit the 3 competing models: gbc, lg, mlp
    print('\nFitting the models...')
    ts = time()
    gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=500, random_state=2,
                                    min_samples_leaf=50, max_depth=5, max_features=3)
    gbc.fit(X_model, y_model)

    lg = LogisticRegression(solver='lbfgs')
    lg.fit(X_std, y_model)

    mlp = define_mlp_model(n_input)
    mlp.fit(X_std, y_model, epochs=30, batch_size=2048, verbose=2, class_weight=weights)
    te= time()
    print("Time passed:", te-ts) 

    # Calculate model_profits for each model and plot
    print('\nCalculating model profits...')
    ts = time()
    models = [gbc, lg, mlp]
    model_profits = []
    for model in models:
        if model == gbc: 
            predicted_probs = model.predict_proba(X_holdout)[:, 1]
        elif model == lg:
            predicted_probs = model.predict_proba(X_holdout_std)[:, 1]
        else:
            predicted_probs = model.predict_proba(X_holdout_std)
        profits, thresholds = profit_curve(cost_benefit, predicted_probs, y_holdout)
        model_profits.append((model, profits, thresholds))
    plot_model_profits(model_profits, 'images/profit_curve.png')
    te= time()
    print("Time passed:", te-ts) 

    # Find the max profit and corresponding model and threshold
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
