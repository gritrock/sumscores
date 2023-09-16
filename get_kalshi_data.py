import pandas as pd
import matplotlib.pyplot as plt
from kalshi_python import Configuration, ApiInstance, MarketApi
from datetime import datetime
import numpy as np
from datetime import datetime, timezone
import time
from tqdm import tqdm
from pprint import pprint
import json
from tqdm import tqdm



# Create a configuration object
config = Configuration()

# Create an API instance passing your email and password

kalshi_api = ApiInstance(email='email@email.com', password='password', configuration=config)




def run_api(api_name, ticker,drop_no_volume=True):
    """wrapper for kalshi API, automatically uses the cursor to get all the results.
    
    Parameters
    ----------
    api_name:str: can be "get_events","get_event","get_market_history"
    ticker: str: refers to ticker of the series/event/market
    drop_no_volume: bool: only applies for get_market_history, says whether or not to drop rows where no trades took place

    
    Returns
    -------
    pandas.DataFrame
    
    Notes
    -----
    For get_market_history volume is originally represented as sum of orders so far, we take the difference between
        sequential rows to change this to the amount of orders in this row
    Uses fill_strike_type_based_on_subtitle to clean up strike values returned when api_name is get_event 
    
    The ticker for get_events looks like "NASDAQ100Y" or "GOLD"
    The ticker for get_event looks like  "NASDAQ100Y-23DEC29" or "GOLD-22SEP1416"
    The ticker for get_market_history looks like "NASDAQ100Y-23DEC29-T8500" or "GOLD-22SEP1416-T750"
    """
    if api_name not in ['get_events','get_event','get_market_history']:
        raise KeyError('api_name must be either get_events or get_event or get_market_history')
        
    
    dfs = []
    i = 0
    cursor = None
    while True:
                
        if api_name=='get_events':
            result = kalshi_api.get_events(cursor=cursor, series_ticker=ticker)
            df = pd.DataFrame([vars(d) for d in result.events])
            df['_series_ticker'] = ticker

            dfs.append(df)

        elif api_name=='get_event':
            result = kalshi_api.get_event( ticker)
            df = pd.DataFrame([vars(d) for d in result.markets])
            df['_event_ticker'] = ticker
            df = df.rename({'_ticker':'market_ticker'}, axis='columns')

            dfs.append(df)
            # this one doesnt have a cursor, need to just break out
            break
            
        elif api_name=='get_market_history':
            result = kalshi_api.get_market_history(ticker,cursor=cursor)
            df = pd.DataFrame([vars(d) for d in result.history])
            df['_market_ticker']=ticker

            # volume is strictly increasing, want volume to represent how much was bought/sold at this row
            volume_change=(df['_volume']-df['_volume'].shift(1))
            volume_change[0]=df['_volume'].values[0]
            df['_volume'] = volume_change


            # dont care if there were no trades
            if drop_no_volume:
                df =df[df['_volume']!=0]

            dfs.append(df)
        
        # need to assign this to get the next batch of results
        cursor = result.cursor
        
        # can stop now
        if df.shape[0]==0 or cursor=='' or cursor is None:
            break
        
    df_all = pd.concat(dfs)
    
    # remove the _ before the column name
    df_all.columns = [c[1:] if c.startswith('_') else c for c in df_all.columns ]

    #fix dat shitty data
    if api_name=="get_event":
        df_all = fill_strike_type_based_on_subtitle(df_all)

    return df_all


def get_all_events_in_multiple_series(series_tickers):
    """
    This function retrieves all events for a list of given series tickers.

    Parameters:
    series_tickers (list of str): A list of series tickers to retrieve events for.

    Returns:
    DataFrame: A DataFrame containing all the events for the given series tickers.
    """
    dfs = []
    for series in tqdm(series_tickers):
        df_event = run_api("get_events",series)
        dfs.append(df_event)
    df_events = pd.concat(dfs)
    return df_events


def get_all_market_history(event_ticker,drop_no_volume=True):
    """
    This function retrieves all market history for a given event ticker.

    Parameters:
    event_ticker (str): The event ticker to retrieve market history for.
    drop_no_volume (bool): says whether or not to drop market history data with no volume
    
    Returns:
    DataFrame: A DataFrame containing all the market history for the given event ticker, 
    along with additional metadata like the strike types, cap strike, floor strike, custom strike, 
    subtitle, market ticker, and status.
    """

    # this gives all the markets for a particular event
    df_market = run_api('get_event', ticker=event_ticker)
    
    histories = []
    for market_ticker in tqdm(df_market.market_ticker.values):
        df_ = run_api('get_market_history', ticker=market_ticker, drop_no_volume=drop_no_volume)
        histories.append(df_)
    
    df_market_history = pd.concat(histories)
    
    # join in metadata about each market, like where the bins are and if they are open/closed
    event_cols = ['strike_type','cap_strike','floor_strike','custom_strike','subtitle','market_ticker','status']
    df_market_history_meta = pd.merge(df_market_history, df_market[event_cols], how='left', on='market_ticker')



    # convert to pandas timestamps
    df_market_history_meta['time_stamp']= (pd.to_timedelta(df_market_history_meta['ts'], unit='s') + pd.to_datetime('1970-1-1'))
    df_market_history_meta['date'] = df_market_history_meta['time_stamp'].dt.date
    df_market_history_meta['date'] = pd.to_datetime(df_market_history_meta['date'], format='%Y-%m-%d')
                                 
    df_market_history_meta = df_market_history_meta.sort_values(['time_stamp', 'market_ticker'],ascending=[True,True])

    # dont want duplicate indexes
    df_market_history_meta.reset_index(inplace=True,drop=True)

    return df_market_history_meta


# ---------- CLEAN EVENT DATA FROM API ---------------

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def subtitle_to_strike(subtitle):
    """helper function for cleaning up api return when using "get_event", fills in strike_type,floor_strike,cap_strike"""

    try:
        subtitle = subtitle.replace('%','').replace('$','').replace(',','')
        if subtitle.startswith('>'):
            subtitle = subtitle.replace('>','')
            if '::' in subtitle: # handle case ">4.75 :: 100 BP+ hike"
                subtitle = subtitle.split('::')[0]
            
            lower = subtitle
            if is_number(lower):
                strike_type='greater'
                floor_strike= float(lower)
                cap_strike=None
            else:
                strike_type=None
                floor_strike= None
                cap_strike=None
                print(subtitle)
        elif 'or higher' in subtitle:
            low = subtitle.split('or higher')[0]
            strike_type='greater'
            floor_strike= float(low)
            cap_strike=None
        elif '<' in subtitle:
            strike_type='less'
            floor_strike= None
            cap_strike=float(subtitle.replace('<',''))
        elif 'or lower' in subtitle:
            high = subtitle.split('or lower')[0]
            strike_type='less'
            floor_strike= None
            cap_strike=float(high)
        elif '-' in subtitle and 'Between' not in subtitle:
            low,high = subtitle.split('-')
            strike_type='between'
            floor_strike= float(low)
            cap_strike=float(high)
        elif 'to'in subtitle:
            low,high = subtitle.split('to')
            strike_type='between'
            floor_strike= float(low)
            cap_strike=float(high)
        elif 'Between' in subtitle:
            subtitle = subtitle.replace('Between ','')
            if 'and' in subtitle:
                low,high = subtitle.split('and')
                strike_type='between'
                floor_strike= float(low)
                cap_strike=float(high)
            if '-' in subtitle:
                low,high = subtitle.split('-')
                strike_type='between'
                floor_strike= float(low)
                cap_strike=float(high)
                
        else:
            strike_type,floor_strike,cap_strike = None,None,None
            print('missing this case')
            print(subtitle)

        return strike_type,floor_strike,cap_strike
    
    except:
        return None,None,None


def fill_strike_type_based_on_subtitle(df):
    """fill/clean in bad api results from get_event api"""

    for idx, row in df.iterrows():
        if pd.isna(row['strike_type']):
            strike_type, floor_strike, cap_strike = subtitle_to_strike(row['subtitle'])
            df.at[idx, 'strike_type'] = strike_type
            df.at[idx, 'floor_strike'] = floor_strike
            df.at[idx, 'cap_strike'] = cap_strike
    
    return df
