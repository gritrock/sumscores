import streamlit as st
import pandas as pd
from kalshi_python import Configuration, ApiInstance, MarketApi
from datetime import datetime
import numpy as np
from datetime import datetime, timezone
import time
from tqdm import tqdm
import json
from tqdm import tqdm
import os 
from get_kalshi_data import *


def load_event_data(event_ticker, policy='bin_edge'):
    df_event = pd.read_csv(f'data/kalshi_events/{event_ticker}.csv')
    df_event = df_event[df_event['volume']!=0]
    df_event = df_event.reset_index(drop=True)
    # # convert to pandas timestamps
    df_event['time_stamp']= (pd.to_timedelta(df_event['ts'], unit='s') + pd.to_datetime('1970-1-1'))
    df_event['date'] = df_event['time_stamp'].dt.date
    df_event['event_ticker'] = event_ticker

    # add interval for each market
    df_event['market_range'] = df_event.apply(lambda row: get_range(row), axis=1)
    df_event = set_midpoint(df_event, policy=policy)
    return df_event


def update_window(df_now, row):
    """
    Purpose: df_now is meant to keep a record of the latest transactions across all the markets in an event
        update_window replaces rows where there is a new record for the market, otherwise it adds the row to df_now
        
    Note row is a row of a pandas dataframe with the same columns as df_now
    """
    
    # replace the row in df_now that has that market_ticker of row['market_ticker'] with row
    if row['market_ticker'] in df_now['market_ticker'].values:
            
        # Convert row into DataFrame and transpose it, reset its index
        row_df = pd.DataFrame(row).T.reset_index(drop=True)

        # Set the index of row_df to match that of the row to update in the DataFrame
        idx = df_now[df_now['market_ticker'] == row['market_ticker']].index
        row_df.index = idx

        # Use DataFrame.update to replace the values
        df_now.update(row_df)

    # add row to df_now
    else:
        df_now.loc[len(df_now)+1] = row
    
    return df_now


    
def aggregate(df_now):
    """Purpose: aggregate data across all markets in an event for a given time slice."""
    
    probs = (df_now['yes_price']/df_now['yes_price'].sum())
    vol_weighted_probs = (df_now['yes_price']*df_now['volume'])/(df_now['yes_price']*df_now['volume']).sum()
    
    expected_value = (df_now['midpoint']*probs).sum()
    vol_weighted_expected_value = (df_now['midpoint']*vol_weighted_probs).sum()

    variance = ((expected_value - df_now['midpoint'])*(expected_value - df_now['midpoint'])*probs).sum()
    std_dev = np.sqrt(variance)
    
    vol_weighted_variance = ((vol_weighted_expected_value - df_now['midpoint'])*(vol_weighted_expected_value - df_now['midpoint'])*vol_weighted_probs).sum()
    vol_weighted_std_dev = np.sqrt(vol_weighted_variance)
    
    
    return pd.Series(
            {
                'time_stamp': df_now.time_stamp.max(),
                'expected_value':expected_value,
                'std_dev':std_dev,
                'yes_price_sum':df_now['yes_price'].sum(),
                'yes_ask_sum':df_now['yes_ask'].sum(),
                'yes_bid_sum': df_now['yes_bid'].sum(),
                'num_markets': df_now.shape[0],
                'vol_weighted_std_dev':vol_weighted_std_dev,
                'vol_weighted_expected_value':vol_weighted_expected_value
                
                
            }
        )


def getmid(interval):
    if interval.left<0:
        return interval.right 
    elif interval.right>10**10:
        return interval.left
    else:
        return (interval.left+interval.right)/2

# TODO add more binning policies
def set_midpoint(df, policy='bin_edge'):
    """Purpose: given the 'market_range' of all the markets in this event
            give the midpoint of each market

    T
    Parameters
    policy - 'bin_edge' - take the bin edge if its the biggest/smallest bin and "open"

    """

    # get the bins in order so we can grab the next bin
    bins = df['market_range'].unique()
    bins = sorted(bins)
    
    midpoints = []
    for i,row in df.iterrows():
        interval = row['market_range']
        if np.isfinite(interval.left) and np.isfinite(interval.right):
            midpoints.append((interval.left +interval.right)/2.)

        # less value or less 
        elif not np.isfinite(interval.left):
            bin_idx = bins.index(interval)
            closest_bin = bins[bin_idx+1]
            # this is the lowest value bin
            if bin_idx == 0:
                if policy=='bin_edge':
                    midpoints.append(interval.right)
            else:
                if np.isfinite(closest_bin.right):
                    new_mid = (closest_bin.right + interval.right)/2 
                else: 
                    new_mid = (closest_bin.left + interval.right)/2 

                midpoints.append(new_mid)

        # this value or bigger
        elif not np.isfinite(interval.right):
            bin_idx = bins.index(interval)
            closest_bin =  bins[bin_idx-1]

            # this is the biggest bin, cant look at next one
            if bin_idx == len(bins)-1:
                if policy=='bin_edge':
                    midpoints.append(interval.left)
            else:
                if np.isfinite(closest_bin.right):
                    new_mid = (closest_bin.right + interval.left)/2 
                else:
                    new_mid = (closest_bin.left + interval.left)/2 

                midpoints.append(new_mid)

    df['midpoint'] = midpoints
    
    return df


def get_range(row):
    if row['strike_type']=='between':
        return pd.Interval(row['floor_strike'],row['cap_strike'])
    elif row['strike_type']=='greater':
        return pd.Interval(row['floor_strike'],np.inf)
    
    elif row['strike_type']=='less':
        return pd.Interval(-np.inf,row['cap_strike'])



            