import pandas as pd
from services.api_handler import APIHandler
from services.tech_analyser import TechAnalyser
from utils.formatter import Formatter
import sys
import argparse
from utils.loader import load_config
from utils.saver import df_to_csv

'''Fetch data'''
'''YAML FETCH_EOD STRUCTURE:
#tickers
VNINDEX: []
VN30: []
#Indicators: sma, ema, rsi, macd, atr, bollinger bands, obv, vwap, cci, mfi, schotastic oscillator, adx, fibonaci retracement
INDICATORS_ALL: []

#Fetch range
fetch_range:
  csv_name: ""
  csv_directory: ""
  start_date: ''
  end_date: ''
  indicators: []
  tickers: []
  date_format: "%d-%m-%Y"
'''
def api_fetch_eod(args):
    df_all = pd.DataFrame()
    config = load_config(args)
    eod_config = config["fetch_range"]
    handler = APIHandler()
    formatter = Formatter()
    resoluton = "1D"
    from_ts = formatter.to_timestamp(eod_config['start_date'], eod_config['date_format'])
    to_ts = formatter.to_timestamp(eod_config['end_date'], eod_config['date_format'])
    tickers = eod_config['tickers']
    tickers = sorted(tickers)
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        handler.fetch_history(from_ts=from_ts, to_ts=to_ts, ticker=ticker, resolution=resoluton)
        tech_analyser = TechAnalyser()
        tech_analyser.add_col_by_list(eod_config['indicators'])
        df_all = pd.concat([df_all, tech_analyser.df], ignore_index=True)
    df_to_csv(df_all, eod_config["csv_directory"] + eod_config["csv_name"])

'''Test API methods'''

def test_api_handler_realtime():
    resolution = "10"
    range = pd.Timedelta(2, "d")
    ticker = "VIC"
    handler = APIHandler()
    handler.fetch_realtime_data(resolution, range, ticker)

def test_api_handler_eod(args):
    handler = APIHandler()
    formatter = Formatter()
    resoluton = "1D"
    from_ts = formatter.to_timestamp("01-01-2025", date_format="%d-%m-%Y")
    to_ts = formatter.to_timestamp("22-09-2025", date_format="%d-%m-%Y")
    ticker = "HPG"
    handler.fetch_history(from_ts=from_ts, to_ts=to_ts, ticker=ticker, resolution=resoluton)

'''Main'''

if __name__ == "__main__":
    tasks = {
        "test_eod": test_api_handler_eod,
        "fetch_eod": api_fetch_eod,
    }
    parser = argparse.ArgumentParser(description="Run specific tasks with settings, other configs are loaded from /configs/")
    parser.add_argument(
        "task",
        type=str,
        help=f"Task to run. Available tasks: {', '.join(tasks.keys())}"
    )
    args = parser.parse_args()
    if args.task not in tasks:
        print(f"Invalid task '{args.task}'")
        print("Available tasks:", ", ".join(tasks.keys()))
        sys.exit(1)
    tasks[args.task](args)
