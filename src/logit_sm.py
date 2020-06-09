import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.final_model import load_split_data

(X_model, X_holdout, y_model, y_holdout), col_names = load_split_data()

## Fit final Logistic model in statsmodel, get the coefficients and p-value
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_std = scaler.fit_transform(X_model)
X_std = sm.add_constant(X_std)

lg = sm.Logit(y_model, X_std)
result = lg.fit(method='lbfgs')
print(result.summary2())
