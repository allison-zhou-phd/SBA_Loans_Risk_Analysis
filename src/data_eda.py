import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import folium

if __name__ == "__main__":
    df = pd.read_pickle('data/pickled_loan')
    df_state = df.groupby('State').mean()['Default']*100

    # df_state.columns =['State', 'Default_Rate']
    # url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
    # state_geo = f'{url}/us-states.json'

    # m = folium.Map(location=[48, -102], zoom_start=3)

    # folium.Choropleth(
    #     geo_data=state_geo,
    #     name='choropleth',
    #     data=df_state,
    #     columns=['State', 'Default Rate'],
    #     key_on='feature.id',
    #     fill_color='YlGn',
    #     fill_opacity=0.7,
    #     line_opacity=0.2,
    #     legend_name='Default Rate (%)'
    # ).add_to(m)

    # folium.LayerControl().add_to(m)

    # m