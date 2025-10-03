import pandas as pd
import numpy as np
import sys
import argparse

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
def task_fetch_history(args):
    from services.tech_analyser import TechAnalyser
    from services.api_handler import APIHandler
    from utils.formatter import Formatter
    from utils.loader import load_config
    from utils.saver import df_to_csv
    df_all = pd.DataFrame()
    config = load_config(args)
    eod_config = config["fetch_range"]
    handler = APIHandler()
    formatter = Formatter()
    resoluton = eod_config["resolution"]
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

'''Training'''
def task_generate_time_series(args):
    from models.simple_lstm import SimpleLSTM
    from services.data_processor import TimeSeriesProcessor
    time_series_processor = TimeSeriesProcessor("simple_lstm")

'''Test methods'''

def task_test_api_handler_realtime():
    from services.api_handler import APIHandler
    from utils.formatter import Formatter
    resolution = "10"   
    range = pd.Timedelta(2, "d")
    ticker = "VIC"
    handler = APIHandler()
    handler.fetch_realtime_data(resolution, range, ticker)

def task_test_api_handler_eod(args):
    from services.api_handler import APIHandler
    from utils.formatter import Formatter
    handler = APIHandler()
    formatter = Formatter()
    resoluton = "1D"
    from_ts = formatter.to_timestamp("01-01-2025", date_format="%d-%m-%Y")
    to_ts = formatter.to_timestamp("22-09-2025", date_format="%d-%m-%Y")
    ticker = "HPG"
    handler.fetch_history(from_ts=from_ts, to_ts=to_ts, ticker=ticker, resolution=resoluton)


def task_test_train_simple_lstm(args):
    from models.simple_lstm import SimpleLSTM
    from torch.utils.data import DataLoader, TensorDataset
    from services.data_processor import TimeSeriesProcessor

def task_test_train_simple_linear(args):
    # Create simple dataset y = x*2
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    from models.simple_linear import SimpleLinear
    X = np.random.rand(1000, 1).astype(np.float32)
    y = (X * 2).astype(np.float32)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    model = SimpleLinear()
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model.trainn(train_loader, val_loader)
'''Main'''

if __name__ == "__main__":
    tasks = {
        "test_eod": task_test_api_handler_eod,           
        "fetch_history": task_fetch_history,
        "test_time_series": task_test_time_series,
        "test_train_simple_lstm": task_test_train_simple_lstm,
        "test_train_simple_linear": task_test_train_simple_linear
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
