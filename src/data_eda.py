import numpy as np
import pandas as pd
import json
import folium
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})


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
        ax.set_xlabel('')
    else:
        df_plot = df_grouped[df_grouped['LoanNr']>cutoff].sort_values('Default', ascending=False)
        ax = df_plot['Default'].plot.hist(bins=bins, alpha=0.8)
        ax.set_title(title)

if __name__ == "__main__":

    df = pd.read_pickle('data/pickled_loan')
    
    us_unemploy = pd.read_csv('data/us_unemployment.csv', index_col=0)
    ur = us_unemploy.values.reshape(-1,1)[:-8]
    date_range = pd.date_range('1965-01','2020-05', freq='M')
    df_ur = pd.DataFrame(data=ur, index=date_range, columns=['U_rate'])
    fig, ax = plt.subplots(figsize=(8,4))
    plt.plot(df_ur)
    ax.grid()
    ax.set_title("U.S. Unemployment Rate (1965.01-2020.04)")
    plt.savefig('images/unemployment.png')
    plt.close()

    # df_state = df.groupby('State').mean()['Default']*100
    # columns =['State', 'Default_Rate']
    # m = heatmap_state(df_state, columns, 'Default Rate (%)', 'YlGn')

    plot_default_rate(df, 'Bank', "Loan Default Rate by Bank (with 3000+ loans)", cutoff=3000)
    plt.savefig('images/default_bank_3000.png')
    plt.close()

    plot_default_rate(df, 'Bank', "Loan Default Rate by Bank (with 1000+ loans)", cutoff = 1000)
    plt.savefig('images/default_bank_1000.png')
    plt.close()

    plot_default_rate(df, 'Sector', "Default Rate by Sector (top 10)")
    plt.savefig('images/default_sector.png')
    plt.close()