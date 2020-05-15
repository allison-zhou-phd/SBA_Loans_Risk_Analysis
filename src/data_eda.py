import numpy as np
import pandas as pd
import json
import folium
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

def heatmap_state(df, cols, legend, color='YlGn'):
    """ 
    Heat-map to plot variable by the U.S. states.
    
    Args:
        df(dataframe): pandas dataframe that contains the data to plot
        col(list): the col name for which the map should be plotted, col[0] contains states code 
        legend(str): legend to use 
        color(str): color theme to use (see https://github.com/dsc/colorbrewer-python)
    Returns:
        Folium interactive map
    """

    url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
    state_geo = f'{url}/us-states.json'

    m = folium.Map(location=[48, -102], zoom_start=3)
    folium.Choropleth(
        geo_data=state_geo,
        name='choropleth',
        data=df,
        columns=cols,
        key_on='feature.id',
        fill_color=color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m

def plot_default_rate(df, col, title, bins=25, cutoff=0):
    """
    Histogram plots of loan default rate by a criterion.  

    Args:
        df(dataframe): pandas dataframe that contains the data to plot
        col(str): column header to group by and plot
        cutoff(int): minimum number of loans needed for the category to be considered
        title(str): plot title
        bins(int): number of bins to use for histogram
    Returns:
        histogram plot
    """
    df_grouped = df.groupby(col).agg({'LoanNr': 'count', 'Default': 'mean'})
    if cutoff==0:
        df_plot = df_grouped.sort_values('Default', ascending=False)[0:10]
        ax = df_plot['Default'].plot.bar(width=0.8)
        ax.set_title(title)
        #ax.set_xlabel('')
    else:
        df_plot = df_grouped[df_grouped['LoanNr']>cutoff].sort_values('Default', ascending=False)
        ax = df_plot['Default'].plot.hist(bins=bins)
        ax.set_title(title)
        ax.set_xlabel('Default Rate')

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

    # us_unemploy = pd.read_csv('data/us_unemployment.csv', index_col=0)
    # ur = us_unemploy.values.reshape(-1,1)[:-8]
    # date_range = pd.date_range('1965-01','2020-05', freq='M')
    # df_ur = pd.DataFrame(data=ur, index=date_range, columns=['U_rate'])
    # fig, ax = plt.subplots(figsize=(8,4))
    # plt.plot(df_ur)
    # ax.grid()
    # ax.set_title("U.S. Unemployment Rate (1965.01-2020.04)")
    # plt.savefig('images/unemployment.png')
    # plt.close()

    df = pd.read_pickle('data/pickled_loan')
    df.drop(['grouper','Date'], axis=1, inplace=True)

    # df_state = df.groupby('State').mean()['Default']*100
    # columns =['State', 'Default_Rate']
    # m = heatmap_state(df_state, columns, 'Default Rate (%)', 'YlGn')

    plot_default_rate(df, 'Sector', "Default Rate by Sector (top 10)")
    plt.savefig('images/default_sector.png')
    plt.close()

    plot_default_rate(df, 'Bank', "Loan Default Rate by Bank Histogram (3000+ loans)", cutoff=3000, bins=10)
    plt.savefig('images/default_bank_3000.png')
    plt.close()

    plot_default_rate(df, 'Bank', "Loan Default Rate by Bank Histogram (1000+ loans)", cutoff=1000, bins=10)
    plt.savefig('images/default_bank_1000.png')
    plt.close()

    # df['SBA_g'] = df['SBA_Appv']/df['GrAppv']
    # df['StateRisk'] = df['State'].map({'FL':2, 'GA':2, 'DC':2, 'NV':2, 'IL':2, 'MI':2, 'TN':2, 'AZ':2, 'NJ':2, 'SC':2,
    #                                    'NY':2, 'MD':2, 'KY':2, 'NC':2, 'TX':2, 'VA':2, 'CA':2, 'LA':1, 'DE':1, 'CO':1, 
    #                                    'UT':1, 'IN':1, 'AR':1, 'AL':1, 'OH':1, 'WV':1, 'MS':1, 'OK':1, 'OR':1, 'HI':1,
    #                                    'MO':1, 'PA':1, 'ID':1, 'CT':1, 'WA':1, 'MA':1, 'KS':1, 'WI':1, 'MN':1, 'IA':1,
    #                                    'RI':1, 'AK':1, 'NE':1, 'NM':1, 'NH':1, 'ME':0, 'SD':0, 'ND':0, 'VT':0, 'WY':0,
    #                                    'MT':0})
    # df['SectorRisk'] = df['Sector'].map({53:2, 52:2, 48:2, 51:2, 61:2, 56:2, 45:2, 23:2, 49:2, 44:2, 
    #                                      72:2, 71:2, 81:1, 42:1, 31:1, 54:1, 32:1, 92:1, 22:1, 33:1,
    #                                      62:1, 55:1, 11:0, 21:0, 0:0})
    
    # df_loan = df[['StateRisk', 'SectorRisk', 'Term', 'NumEmp','LowDocu', 'GrAppv', 'SBA_g', 'U_rate','Default']]
    # df_loan.to_pickle('data/loan_data')   

    # X1 = df_loan[['StateRisk', 'SectorRisk', 'Term', 'NumEmp','GrAppv', 'SBA_g', 'U_rate']].values
    # X2 = df_loan[['LowDocu']].values
    # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # X1_std_ss = scaler.fit_transform(X1)
    # X = np.hstack((X1_std_ss, X2))
    # pca = PCA(n_components=8)
    # X_pca = pca.fit_transform(X)

    # fig, ax = plt.subplots(figsize=(8,4))
    # plot_pca_explained_var(ax, pca)
    # plt.savefig('images/PCA_exaplained_var.png')
    # plt.close()