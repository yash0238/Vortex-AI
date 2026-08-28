import yfinance as yf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","KOTAKBANK.NS","BHARTIARTL.NS","ITC.NS","AXISBANK.NS",
    "SBIN.NS","LT.NS","BAJFINANCE.NS","HCLTECH.NS","ASIANPAINT.NS",
    "MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS",
    "WIPRO.NS","POWERGRID.NS","NTPC.NS","M&M.NS","TECHM.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS","BAJAJ-AUTO.NS","CIPLA.NS",
    "DRREDDY.NS","DIVISLAB.NS","HEROMOTOCO.NS","ONGC.NS","COALINDIA.NS",
    "BPCL.NS","GRASIM.NS","ADANIPORTS.NS","EICHERMOT.NS","APOLLOHOSP.NS",
    "HINDALCO.NS","TATACONSUM.NS","BRITANNIA.NS","SHREECEM.NS","UPL.NS",
    "BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS","INDUSINDBK.NS","LTI.NS"
]

def download_all():
    data = yf.download(NIFTY50, start="2010-01-01", end="2024-12-31",
                        auto_adjust=True)
    close = data["Close"].dropna(how="all")
    volume = data["Volume"].dropna(how="all")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    close.to_csv(RAW_DATA_DIR / "nifty50_close.csv")
    volume.to_csv(RAW_DATA_DIR / "nifty50_volume.csv")
    print(f"Downloaded: {close.shape[0]} days x {close.shape[1]} stocks")
    return close, volume

if __name__ == "__main__":
    download_all()
