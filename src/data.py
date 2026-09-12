import pandas as pd
import numpy as np
import yfinance as yf
import torch
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

print("Number of stocks:", len(stocks))

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("yfinance:", yf.__version__)
print("torch:", torch.__version__)
# print("PyG:", torch_geometric.__version__)
print("CUDA available:", torch.cuda.is_available())

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

# df for 2024-09-11 date 
df11 = pd.DataFrame(raw_data['2024-09-11'])
print()