from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
import json

root = Path(__file__).resolve().parent.parent

class TechAnalyser:
    df = pd.DataFrame()

    def load_json(self, relative_path: str) -> dict:
        """Đọc file JSON và trả về dict"""
        json_path = root / relative_path
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    
    def data_processor(self, path):
        json_path = root / path 
        #take data from json file

        data = self.load_json(json_path)
        
        #process data to dataframe
        ticker = data["params"]["symbol"]
        #ticker column must represent ticker in all rows
        tickers = [ticker] * len(data["t"])
        self.df["ticker"] = tickers
        self.df["timestamp"] = pd.to_datetime(data["t"], unit='s')
        self.df["open"] =  data["o"]
        self.df["high"] =  data["h"]
        self.df["low"] =  data["l"]
        self.df["close"] =  data["c"]
        self.df["volume"] =  data["v"]    

    def sma(self, period: int):
        self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
    
    def ema(self, period: int):
        self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
    
    def rsi(self, period: int = 14):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    def macd(self, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        ema_short = self.df['close'].ewm(span=short_period, adjust=False).mean()
        ema_long = self.df['close'].ewm(span=long_period, adjust=False).mean()
        self.df['macd'] = ema_short - ema_long
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal_period, adjust=False).mean()

    def atr(self, period: int = 14):
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df[f'atr_{period}'] = tr.rolling(window=period).mean()

    def bollinger_band(self, period: int = 20, num_std_dev: int = 2):
        sma = self.df['close'].rolling(window=period).mean()
        std_dev = self.df['close'].rolling(window=period).std()
        self.df[f'bollinger_lband_{period}'] = sma - (std_dev * num_std_dev)
        self.df[f'bollinger_hband_{period}'] = sma + (std_dev * num_std_dev)



    def test(self):
        path = "data/json/cache.json"
        self.data_processor(path)
        self.ema(9)
        self.ema(26)
        self.rsi(14)
        self.sma(5)
        self.sma(20)
        self.sma(50)
        self.
        print(self.df)
        self.df.to_csv("data/csv/temp.csv", index=False)
    


if __name__ == "__main__":
    analyser = TechAnalyser()
    analyser.test()
        





