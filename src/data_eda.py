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

def plot_default_rate(df, col, title, sector_dict=None, ascending=True, bins=25, cutoff=0):
    """
    Plots of loan default rate by a criterion.  If a cutoff level is given, a histogram is plotted
    of the data meeting the cutoff.  Otherwise a bar plot is generated with decreasing default rates.

    Args:
        df(dataframe): pandas dataframe that contains the data to plot
        col(str): column header to group by and plot
        title(str): plot title
        bins(int): number of bins to use for histogram
        cutoff(int): minimum number of loans needed for the data to be considered  
    Returns:
        histogram plot / bar plot
    """

    df_grouped = df.groupby(col).agg({'LoanNr': 'count', 'Default': 'mean'})
    if cutoff==0:
        if ascending:
            df_plot = df_grouped.sort_values('Default', ascending=ascending)[1:6]
            df_plot = df_plot.sort_values('Default', ascending=False)
            x_labels = []
            for i in df_plot.index:
                x_labels.append(sector_dict[i])
            ax = df_plot['Default'].plot.bar(width=0.7)
            ax.set_title(title)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
            ax.set_ylim(0, 0.3)
            ax.set_xlabel('Sector')
            #ax.set_yticklabels('')
            ax.set_ylabel('Default Rate')
        else:
            df_plot = df_grouped.sort_values('Default', ascending=ascending)[0:5]
            x_labels = []
            for i in df_plot.index:
                x_labels.append(sector_dict[i])
            ax = df_plot['Default'].plot.bar(width=0.7)
            ax.set_title(title)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
            ax.set_ylim(0, 0.3)
            ax.set_xlabel('Sector')
            ax.set_ylabel('Default Rate')
    else:
        df_plot = df_grouped[df_grouped['LoanNr']>cutoff].sort_values('Default', ascending=False)
        ax = df_plot['Default'].plot.hist(bins=bins)
        ax.set_title(title)
        ax.set_xlabel('Default Rate')
        ax.set_ylabel('Frequency')

def plot_pca_explained_var(ax, pca):
    """
    Plots the percentage of total var explained by each additional pca.

    Args:
        ax: axis on which to plot
        pca: the principal components
    Returns:
        A line plot with a horizontal line at 90%
    """
    
    total_var = np.sum(pca.explained_variance_)
    cumsum_var = np.cumsum(pca.explained_variance_)
    prop_var_expl = cumsum_var/total_var
    ax.plot(prop_var_expl, color = 'black', linewidth=2, label='Explained variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', linewidth=1)
    ax.set_ylabel('proportion of explained variance')
    ax.set_xlabel('number of principal components')
    ax.legend()

if __name__ == "__main__":

    ### Plot unemployment rate
    unemployment = 0
    if unemployment:
        us_unemploy = pd.read_csv('data/us_unemployment.csv', index_col=0)
        ur = us_unemploy.values.reshape(-1,1)[:-8]
        date_range = pd.date_range('1965-01','2020-05', freq='M')
        df_ur = pd.DataFrame(data=ur, index=date_range, columns=['U_rate'])
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(df_ur)
        ax.set_title("U.S. Unemployment Rate (1965.01-2020.04)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Unemployment Rate")
        plt.tight_layout()
        plt.savefig('images/unemployment.png')
        plt.close()

    df = pd.read_pickle('data/pickled_loan')

    ### Plot loan default rates by states
    state = 0
    if state:
        df_state = df.groupby('State').mean()['Default']*100
        columns =['State', 'Default_Rate']
        m = heatmap_state(df_state, columns, 'Default Rate (%)', 'YlOrRd')

    ### Plot loan default rates by sectors
    sector = 1
    if sector:
        sector_dict = {11: 'Agri, Forest, Fishing',
                    21: 'Mining, Oil & Gas',
                    22: 'Utilities',
                    23: 'Construction',
                    31: 'Manufacturing',
                    32: 'Manufacturing',
                    33: 'Manufacturing',
                    42: 'Wholesale',
                    44: 'Retail',
                    45: 'Retail',
                    48: 'Transportation',
                    49: 'Transportation',
                    51: 'Information',
                    52: 'Finance & Insurance',
                    53: 'Real Estate',
                    54: 'Professional Services',
                    55: 'Management',
                    56: 'Administration and Support',
                    61: 'Education',
                    62: 'Health Care',
                    71: 'Arts & Entertainment',
                    72: 'Accommodation & Food Services',
                    81: 'Other Services',
                    92: 'Public Administration'}

        plot_default_rate(df, 'Sector', "Sectors Most Likely to Default", sector_dict, ascending=False)
        plt.tight_layout()
        plt.savefig('images/default_sector_top5_readme.png')
        plt.close()

        plot_default_rate(df, 'Sector', "Sectors Least Likely to Default", sector_dict, ascending=True)
        plt.tight_layout()
        plt.savefig('images/default_sector_bottom5_readme.png')
        plt.close()

    ### Plot loan default rates by banks
    bank = 0
    if bank:
        plot_default_rate(df, 'Bank', "Loan Default Rate by Bank Histogram (3000+ loans)", cutoff=3000, bins=10)
        plt.savefig('images/default_bank_3000.png')
        plt.close()

        plot_default_rate(df, 'Bank', "Loan Default Rate by Bank Histogram (1000+ loans)", cutoff=1000, bins=10)
        plt.savefig('images/default_bank_1000.png')
        plt.close()

    ### Additional feature engineering and save final data to pickle file
    update_data = 0
    if update_data:
        df['SBA_g'] = df['SBA_Appv']/df['GrAppv']
        df['StateRisk'] = df['State'].map({'FL':2, 'GA':2, 'DC':2, 'NV':2, 'IL':2, 'MI':2, 'TN':2, 'AZ':2, 'NJ':2, 'SC':2,
                                        'NY':2, 'MD':2, 'KY':2, 'NC':2, 'TX':2, 'VA':2, 'CA':2, 'LA':1, 'DE':1, 'CO':1, 
                                        'UT':1, 'IN':1, 'AR':1, 'AL':1, 'OH':1, 'WV':1, 'MS':1, 'OK':1, 'OR':1, 'HI':1,
                                        'MO':1, 'PA':1, 'ID':1, 'CT':1, 'WA':1, 'MA':1, 'KS':1, 'WI':1, 'MN':1, 'IA':1,
                                        'RI':1, 'AK':1, 'NE':1, 'NM':1, 'NH':1, 'ME':0, 'SD':0, 'ND':0, 'VT':0, 'WY':0,
                                        'MT':0})
        df['SectorRisk'] = df['Sector'].map({53:2, 52:2, 48:2, 51:2, 61:2, 56:2, 45:2, 23:2, 49:2, 44:2, 
                                            72:2, 71:2, 81:1, 42:1, 31:1, 54:1, 32:1, 92:1, 22:1, 33:1,
                                            62:1, 55:1, 11:0, 21:0, 0:0})
        
        df_loan = df[['StateRisk', 'SectorRisk', 'Term', 'NumEmp','LowDocu', 'GrAppv', 'SBA_g', 'U_rate','Default']]
        df_loan.to_pickle('data/loan_data')   

    ### PCA analysis
    pca = 0
    if pca:
        X1 = df_loan[['StateRisk', 'SectorRisk', 'Term', 'NumEmp','GrAppv', 'SBA_g', 'U_rate']].values
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