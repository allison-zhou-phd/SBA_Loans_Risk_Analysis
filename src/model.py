import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_pca_explained_var(ax, pca):
    total_var = np.sum(pca.explained_variance_)
    cumsum_var = np.cumsum(pca.explained_variance_)
    prop_var_expl = cumsum_var/total_var
    ax.plot(prop_var_expl, color = 'black', linewidth=2, label='Explained variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', linewidth=1)
    ax.set_ylabel('proportion of explained variance')
    ax.set_xlabel('number of principal components')
    ax.legend()

if __name__ == "__main__":

    df_loan = pd.read_pickle('data/loan_data')
    y = df_loan.pop('Default').values
    X = df_loan.values
    X1 = df_loan[['stateRisk', 'sectorRisk', 'Term', 'NumEmp','GrAppv', 'SBA_g', 'U_rate']].values
    X2 = df_loan[['LowDocu']].values
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X1_std_ss = scaler.fit_transform(X1)
    X = np.hstack((X1_std_ss, X2))
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8,4))
    plot_pca_explained_var(ax, pca)
    plt.savefig('images/PCA_exaplained_var.png')
    plt.close()