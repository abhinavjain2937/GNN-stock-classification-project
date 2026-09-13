import pandas as pd
import numpy as np
import yfinance as yf
import torch

from pathlib import Path

import os 
# import torch_geometric
import sklearn

stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",

    "LT.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
    "BAJFINANCE.NS",
    "HCLTECH.NS",

    "WIPRO.NS",
    "TECHM.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "ONGC.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",

    "TATAMOTORS.NS",
    "M&M.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",

    "BPCL.NS",
    "IOC.NS",
    "GRASIM.NS",
    "BRITANNIA.NS",
    "TATACONSUM.NS",
    "INDUSINDBK.NS",
    "HINDALCO.NS",
    "SHRIRAMFIN.NS",
    "BEL.NS",
    "TRENT.NS"
]



raw_data = yf.download(
    tickers=stocks,
    period='2y',
    interval="1d",
    auto_adjust=True,
    group_by="column",
    progress=True,
    threads=True
    )
print(raw_data.shape)
print(raw_data.head())



# extractig diffrent  data frames
close_df = raw_data['Close']

volume = raw_data['Volume']

high = raw_data['High'] 

low = raw_data['Low']

opening = raw_data['Open']

# removing all the NAN values
close_df = close_df.dropna(axis=1,how='all')
close_df = close_df.dropna(axis=0,how='any')
volume = volume[close_df.columns]
volume = volume.dropna(axis=0,how='any')
high = high[close_df.columns]
high = high.dropna(axis=0,how='any')
low = low[close_df.columns]
low = low.dropna(axis=0,how='any')
opening = opening[close_df.columns]
opening = opening.dropna(axis=0,how='any')

# 1. Daily return
r1 = close_df.pct_change()
print("Returns daily: ",r1)
r5 = close_df.pct_change(5)
r20 = close_df.pct_change(20)


# 3. Volume change
volume_change = volume.rolling(window=20).mean()

# 4. Open-Close return
open_close_return = (close_df - opening) / opening

# 5. Rolling volatility
volatility = r1.rolling(window=20).std()

#6. M20 ratio
M20Ratio = close_df.rolling(window=20).mean()/close_df

# Target
nxt_day_return = r1.shift(-1) 
target = (nxt_day_return > 0).astype(int)

# concatinaation
features_df = pd.concat(
    {

        'R1': r1,
        'R5': r5,
        'R20': r20,
        'M20Ratio': M20Ratio,
        'VolumeChange': volume_change,
        'Volatility': volatility,
        'OpenCloseReturn': open_close_return,
        'Target':target
    },
    axis=1
)
print(close_df.shape)
# 1. Drop rows with missing values
features_df = features_df.dropna()

# 2. Extract your specific date slice and unstack it
date_df = features_df.loc['2026-09-10'].unstack(level='Ticker').T

date_df['Target'] = date_df['Target'].astype(int).astype(str)

# View the result
print(f" the  new data ------ 10 sep 2026:\n{date_df.head()} ")





# Location of this Python file
BASE_DIR = Path(__file__).resolve().parent.parent

# Project data folder
processed_dir = BASE_DIR / "data" / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)

csv_ready_df = date_df.reset_index()
return_csv = r1.reset_index()

output_path = processed_dir / "nifty50_features.csv"
output_path2 = processed_dir / "Return1.csv"

csv_ready_df.to_csv(output_path, index=False)

csv_ready_df.to_csv(output_path2, index=False)


print(f"Processed dataset successfully saved to:")
print(output_path)
print(output_path2)