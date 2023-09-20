
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
from helpers import *

# Create a title for your app
st.title("US Annual inflation simulated data")

# Create a bar across the top of the page with tabs
st.markdown("""
<style>
[data-testid="stHorizontalBlock"] > div {
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# set up your tabs
tab_events, tab_betting_markets, = st.tabs(["📈Events", "Betting Markets"])


# Display different content depending on the selected tab
with tab_events:

    # Create placeholders for content
    content = st.empty()

    # Read CSV files
    df1 = pd.read_csv("randata.csv")
    df2 = pd.read_csv("simulatedpredictions2.csv")

    # Create line chart
    fig = px.line(df1, x="date", y="prediction", title="US annual inflation simulated data")

    # Get the last percentage value and date
    last_value = df1["prediction"].iloc[-1]
    last_date = df1["date"].iloc[-1]

    # Add text annotation
    fig.add_annotation(
        x=last_date,
        y=last_value,
        text=f"{last_value}% live forecast",
        showarrow=True,
        # yshift=10
    )


    for i, row in df2.iterrows():
        fig.add_layout_image(
            dict(
                    source=Image.open("{}.jpg".format(row['source'])),
                    x=row["date"],
                    y=row["prediction"],
                    xref="x",
                    yref="y",
                    sizex=10,
                    sizey=10,
                    xanchor="center",
                    yanchor="middle",
                    # sizing="stretch",
                    layer="above",
                    opacity=.8
                )
            )
    # need this to add hover info over the images 
    # TODO see if can put this into add_layout_image
    fig.add_scatter(
        x=df2["date"],
        y=df2["prediction"],
        mode="markers",
        name="",
        hovertemplate="%{x}<br>%{y}<br>%{text}", # use a single text value
        text=df2.apply(lambda row: f"Source: {row['source']}<br>Score: {row['score']}", axis=1), # combine source and score columns
        hoverinfo="x+y+text" # remove name and list from hover data
    )

    # Display chart in Streamlit app
    content.plotly_chart(fig, use_container_width=True)

with  tab_betting_markets:
    df_series =pd.read_csv('kalshi_series.csv')
    df_events = pd.read_csv('kalshi_events.csv')
    fig, ax = plt.subplots()

    # plot volume per series
    vol_per_series = df_events.groupby('series_ticker')['volume'].sum().sort_values(ascending=False)
    ax = vol_per_series.plot(kind='bar',logy=True)
    ax.set_ylabel('log(volume)')
    st.pyplot(fig)


    series_filter = st.selectbox('Filter By Series', df_series.series_ticker.values, key='series' ,label_visibility="visible")
    if series_filter:
        events = df_events[df_events['series_ticker']==series_filter].event_ticker.values
    else:
        events = df_events.event_ticker.values
    

    event_filter = st.selectbox('Filter By Series', events, index=0, key='events', label_visibility="visible")

    # this is the history of all the markets in the event
    df_event = load_event_data(event_filter)
    
    st.write(f"Total Volume: {df_event.volume.sum()}")
    st.dataframe(df_event, height=200)


    fig = px.line(df_event, x='time_stamp', y='yes_price', color='market_range', title='Price by Market Ticker and Date')
    st.plotly_chart(fig)


    # get the last market for every market before this row, then aggregate it
    df_aggs = []
    df_now = pd.DataFrame(columns=df_event.columns)
    for i,row in df_event.iterrows():
        # this grabs the last market trades  for every market
        df_now = update_window(df_now, row)

        df_ = aggregate(df_now)

# TODO SET PROBS if there are open intervals, handle cases where not all markets have been involved yet.


        # need this if there is only one market
        if isinstance(df_,pd.Series):
            df_ = pd.DataFrame(df_).T

        df_aggs.append(df_)

    # df_agg is an aggregated version of df_event
    df_agg = pd.concat(df_aggs)   
    
    st.dataframe(df_agg)


    # Groupby 'date' and 'market_range' and calculate the mean of 'yes_price'
    df_grouped = df_event.groupby(['date', 'market_range'], as_index=False)['yes_price'].mean()

    # Create a new DataFrame where we'll store all combinations of dates and tickers
    unique_dates = pd.date_range(start=df_grouped['date'].min(), end=df_grouped['date'].max())
    unique_tickers = sorted(df_grouped['market_range'].unique())
    new_df = pd.DataFrame(index=pd.MultiIndex.from_product([unique_dates, unique_tickers], names=['date', 'market_range']))

    # Convert the 'date' column in new_df to the same type as in df_grouped
    new_df.reset_index(inplace=True)
    new_df['date'] = new_df['date'].dt.date

    # Merge our grouped DataFrame with the new one (this will cause NaN values where data was missing)
    merged_df = pd.merge(new_df, df_grouped, on=['date', 'market_range'], how='left')

    # Fill NaNs with the latest available values for each ticker
    merged_df['yes_price'] = merged_df.groupby('market_range')['yes_price'].apply(lambda group: group.ffill())

    # Sort data before creating the heatmap
    merged_df.sort_values(by=['market_range', 'date'], inplace=True)

    merged_df['midpoint']=merged_df['market_range'].apply(lambda x:getmid(x))

    # Now create the heatmap
    pivot_df = merged_df.pivot('midpoint', 'date', 'yes_price')
    pivot_df.index=list(map(float, pivot_df.index))  # convert indices to float instead of str

    # Create figure and add the heatmap
    fig = go.Figure(data=go.Heatmap(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index, colorscale='blues'))

    # Add the scatter plot to the figure
    fig.add_trace(go.Scatter(x=df_agg.time_stamp, y=df_agg.expected_value, mode='lines', name='expected_value',line=dict(color='darkblue')))

    st.plotly_chart(fig)

    # st.table(df_agg)


# with tab_sources:
#     # import required modules
#     from PIL import Image
#     import pandas as pd
#     import os
#     import matplotlib.pyplot as plt
#     import numpy as np

#     def overlay_text_on_image(image, text, output_path, color="black", fontsize=72):
#         img_np = np.array(image)
#         fig, ax = plt.subplots()
#         ax.imshow(img_np)
#         img_width = img_np.shape[1]
#         img_height = img_np.shape[0]
#         ax.text(img_width / 2, img_height / 2, text, fontsize=fontsize, color=color, ha='center', va='center')
#         plt.axis("off")
#         plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
#         plt.close(fig)


#     def create_columns(person_info, name):
#         cols = st.columns(num_columns)

#         cols[1].subheader(str(name))

#         set_markdown(cols[0], person_info, "rank", font_size=50, font_color="green")
#         set_markdown(cols[3], person_info, "genscore", font_size=50, font_color="blue")

#         set_avatar(cols[2], name)
#         set_icon(cols[4], person_info, "predictions", "streak")
#         set_icon(cols[5], person_info, "bankroll", "bankroll")

#         return cols


#     def set_markdown(column, person_info, key, font_size=50, font_color="green"):
#         value_str = str(int(person_info[key]))
#         column.markdown(f"<p style='font-size:{font_size}px; color:{font_color}'>{value_str}</p>", unsafe_allow_html=True)


#     def set_avatar(column, name):
#         avatar_path = os.path.join("data", name + '.jpg')
#         avatar = Image.open(avatar_path)
#         column.image(avatar, width=100)


#     def set_icon(column, person_info, overlay_key, icon_key):
#         # Define icon based on streak value
#         icon_path = get_icon_path(icon_key, person_info)
#         overlay_value = str(int(person_info[overlay_key]))

#         # Overwrite the overlay value onto the icon, save it
#         icon_image = Image.open(icon_path)
#         output_path = os.path.join("data", name + "_" + icon_key + ".jpg")
#         overlay_text_on_image(icon_image, overlay_value, output_path)

#         # Display the icon image
#         column.image(output_path, width=100)


#     def get_icon_path(icon_key, person_info):
#         if icon_key == "streak":
#             return get_streak_icon(person_info['streak'])
#         elif icon_key == "bankroll":
#             return get_bankroll_icon(person_info['bankroll'])


#     def get_streak_icon(streak):
#         if streak > 5:  
#             return "data/hotball.jpg"
#         elif 1 < streak <= 5: 
#             return "data/warmball.jpg"
#         elif -1 <= streak >= -5:
#             return "data/coldball.jpg"
#         elif streak <= -5:
#             return "data/iceball.jpg"


#     def get_bankroll_icon(bankroll):
#         if bankroll > 10000:  
#             return "data/moneybag3.jpg"
#         elif 1 < bankroll <= 10000: 
#             return "data/moneybag2.jpg"
#         elif bankroll <= 1:
#             return "data/moneybag1.jpg"
#         # Main program

#     # After reading the CSV file
#     data = pd.read_csv("sourcesscores.csv", header=0)
#     data = data.set_index("source")

#     # Create a list of names for each person
#     # Create a list of names from the index of the data DataFrame
#     names = data.index.tolist()

#     num_columns = len(data.columns) + 1 # +1 because we're adding a name column

#     # Create a row of columns at the top for the column names
#     header_cols = st.columns(num_columns)

#     # Set the column names
#     header_cols[0].write('RANK')
#     header_cols[1].write('NAME')
#     header_cols[2].write('')
#     header_cols[3].write('GENIUS SCORE')
#     header_cols[4].write('STREAK')
#     header_cols[5].write('BANKROLL')

#     # Loop over the persons
#     for name in names:
#         person_info = data.loc[name]
#         create_columns(person_info, name)



#     with tab_settings:
#     st.write('in tab 4')



# import pandas as pd
# import numpy as np
# import ast
# import openai
# import streamlit as st
# import plotly.express as px
# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
# # Import libraries
# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # Read CSV files
# df1 = pd.read_csv("randata.csv")
# df2 = pd.read_csv("simulatedpredictions2.csv")

# # Create line chart
# fig = px.line(df1, x="Date", y="PredictedRate", title="US annual inflation simulated data") # remove showlegend argument

# # Update layout
# fig.update_layout(showlegend=False) # add this line to hide legend
# # Update layout
# fig.update_layout(hovermode="closest") # add this line to change hover mode

# # Get the last percentage value and date
# last_value = df1["PredictedRate"].iloc[-1]
# last_date = df1["Date"].iloc[-1]
# # 
# # Add text annotation
# fig.add_annotation(
#     x=last_date,
#     y=last_value,
#     text=f"{last_value}% live forecast",
#     showarrow=False,
#     yshift=10
# )

# # Add scatter plot with second data set and hover data
# fig.add_scatter(
#     x=df2["date"],
#     y=df2["prediction"],
#     mode="markers",
#     name="",
#     hovertemplate="%{x}<br>%{y}<br>%{text}", # use a single text value
#     text=df2.apply(lambda row: f"Source: {row['source']}<br>Score: {row['score']}", axis=1), # combine source and score columns
#     hoverinfo="x+y+text" # remove name and list from hover data
# )

# # Update layout
# fig.update_layout(hovermode="closest") # add this line to change hover mode

# # Display chart in Streamlit app
# st.plotly_chart(fig, use_container_width=True)