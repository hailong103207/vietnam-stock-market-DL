import pandas as pd
from services.api_handler import APIHandler
from utils.formatter import Formatter
import sys
import argparse

'''Test API methods'''
def test_api_handler_realtime():
    resolution = "10"
    range = pd.Timedelta(2, "d")
    ticker = "VIC"
    handler = APIHandler()
    handler.fetch_realtime_data(resolution, range, ticker)

def test_api_handler_eod():
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
    }
    parser = argparse.ArgumentParser(description="Run specific tasks with settings.")
    parser.add_argument(
        "task",
        type=str,
        help=f"Task to run. Available tasks: {', '.join(tasks.keys())}"
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default="01-01-2025",
        help="Start date in format dd-mm-yyyy (default: 01-01-2023)"
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default="",
        help="End date in format dd-mm-yyyy (default: today)"
    )
    args = parser.parse_args()
    if args.task not in tasks:
        print(f"Invalid task '{args.task}'")
        print("Available tasks:", ", ".join(tasks.keys()))
        sys.exit(1)
    tasks[args.task](args)