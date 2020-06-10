import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12})

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence

from src.default_modeler import DefaultModeler, undersample, load_split_data

def compare_models(dm_lst, X_train, X_test, y_train, y_test):
    """ 
    Compare models passed in the dm_lst

    Args:
        dm_lst(lst): a list of defaultModler objects
        X_train: ndarray - 2D
        X_test: ndarray - 2D
        y_train: ndarray - 1D
        y_test: ndarray - 1D
    Returns:
        Nothing, the results are printed out
    """
    for dm in dm_lst:
        dm.print_model_metrics(X_train, X_test, y_train, y_test)

if __name__ == "__main__":

    ### X_model and y_model to be used in training the model, the holdout sets for final model evaluation
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data()
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42, stratify=y_model)
    
    ### Rely on the undersample method to balance the data
    undersample = 0
    if undersample:
        target_ratio = 0.45
        X_sampled, y_sampled = undersample(X_model, y_model, target_ratio)

        lg_model = LogisticRegression(solver='lbfgs')
        dm_lg = DefaultModeler(lg_model)
        rfc = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=2)
        dm_rfc = DefaultModeler(rfc)
        gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=300, random_state=2,
                                        min_samples_leaf=200, max_depth=3, max_features=3)
        dm_gbc = DefaultModeler(gbc)
        abc = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', n_estimators=300, random_state=2,
                                        min_samples_leaf=200, max_depth=3, max_features=3)
        dm_abc = DefaultModeler(abc)
        dm_lst = [dm_lg, dm_rfc, dm_gbc, dm_abc]
        compare_models(dm_lst, X_sampled, X_test, y_sampled, y_test)

    ### Rely on class_weight option to balance the data
    classweight = 1
    if classweight: 
        lg = LogisticRegression(solver='lbfgs', class_weight='balanced')
        dm_lg = DefaultModeler(lg)
        rfc = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=2, class_weight='balanced')
        dm_rfc = DefaultModeler(rfc)
        gbc = GradientBoostingClassifier(learning_rate=0.2, n_estimators=300, random_state=2,
                                        min_samples_leaf=200, max_depth=3, max_features=3)
        dm_gbc = DefaultModeler(gbc)
        abc = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', n_estimators=300, random_state=2,
                                        min_samples_leaf=200, max_depth=3, max_features=3)
        dm_abc = DefaultModeler(abc)
        dm_lst = [dm_lg, dm_rfc, dm_gbc, dm_abc]
        compare_models(dm_lst, X_train, X_test, y_train, y_test)
  
    ### Plot feature importance and partial dependence
    plot = 1
    if plot:
        col_labels = np.array(['State Risk', 'Sector Risk', 'Term', 'Number of Employees', 'Low Document', 'Loan Amt', 'SBA Guarantee Ratio', 'Unemployment Rate'])
        dm_rfc.plot_feature_importance(X_train, col_labels)
        plt.savefig('images/rfc_feature_importances.png')
        plt.close()
        dm_gbc.plot_feature_importance(X_train, col_labels)
        plt.savefig('images/gbc_feature_importances.png')
        plt.close()

        features = [2, 7, 6, 5, 1]
        plot_partial_dependence(gbc, X_train, features=features, feature_names=col_labels) 
        fig = plt.gcf()
        fig.set_size_inches(11,8)
        plt.tight_layout()
        plt.savefig('images/gbc_partDepend.png')
        plt.close()


    