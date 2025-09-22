import pandas as pd
from services.api_handler import APIHandler
from utils.formatter import Formatter





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
    from_ts = formatter.to_timestamp("02-01-2014", date_format="%d-%m-%Y")
    to_ts = formatter.to_timestamp("29-04-2014", date_format="%d-%m-%Y")
    ticker = "AAA"
    handler.fetch_history(from_ts=from_ts, to_ts=to_ts, ticker=ticker, resolution=resoluton)

'''Main'''
if __name__ == "__main__":
    test_api_handler_eod()
    