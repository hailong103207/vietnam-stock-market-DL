from abc import ABC,abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
import json

root = Path(__file__).resolve().parent.parent
class BaseAnalyser(ABC):
    @abstractmethod
    def add_sma(self, period: int): pass
    def add_ema(self, period: int): pass
    def add_rsi(self, period: int = 14): pass
    def add_macd(self, short_period: int = 12, long_period: int = 26, signal_period: int = 9): pass
    def add_atr(self, period: int = 14): pass
    def add_bollinger_bands(self, period: int = 20, num_std_dev: int = 2): pass
    def add_obv(self): pass
    def add_vwap(self): pass
    def add_cci(self, period: int = 20): pass
    def add_mfi(self, period: int = 14): pass
    def add_stochastic_oscillator(self, k_period: int = 14, d_period: int = 3): pass
    def add_adx(self, period: int = 14): pass
    def add_fibonacci_retracement(self): pass
    def add_ichimoku_cloud(self, conversion_period: int = 9, base_period: int = 26, span_b_period: int = 52, displacement: int = 26): pass



class TechAnalyser(BaseAnalyser):
    '''
    Indicators: sma, ema, rsi, macd, atr, bollinger bands, obv, vwap, cci, mfi, schotastic oscillator, adx, fibonaci retracement
    '''

    def __init__(self, path: str = "data/json/cache.json", csv_path: str = "data/csv/cache.csv"):
        self.df = pd.DataFrame()
        self.data_preprocess(path)
        self.col_by_str = {
            "SMA": self.add_sma,
            "EMA": self.add_ema,
            "RSI": self.add_rsi,
            "MACD": self.add_macd,
            "ATR": self.add_atr,
            "BOL": self.add_bollinger_bands,
            "OBV": self.add_obv,
            "VWAP": self.add_vwap,
            "CCI": self.add_cci,
            "MFI": self.add_mfi,
            "STO": self.add_stochastic_oscillator,
            "ADX": self.add_adx,
            "FIB": self.add_fibonacci_retracement,
            "ICHI": self.add_ichimoku_cloud
        }

    def load_json(self, relative_path: str) -> dict:
        """Đọc file JSON và trả về dict"""
        json_path = root / relative_path
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def data_preprocess(self, path):
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

    def add_sma(self, period: int):
        self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
    
    def add_ema(self, period: int):
        self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
    
    def add_rsi(self, period: int = 14):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    def add_macd(self, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        ema_short = self.df['close'].ewm(span=short_period, adjust=False).mean()
        ema_long = self.df['close'].ewm(span=long_period, adjust=False).mean()
        self.df['macd'] = ema_short - ema_long
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal_period, adjust=False).mean()

    def add_atr(self, period: int = 14):
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df[f'atr_{period}'] = tr.rolling(window=period).mean()

    def add_bollinger_band(self, period: int = 20, num_std_dev: int = 2):
        sma = self.df['close'].rolling(window=period).mean()
        std_dev = self.df['close'].rolling(window=period).std()
        self.df[f'bollinger_lband_{period}'] = sma - (std_dev * num_std_dev)
        self.df[f'bollinger_hband_{period}'] = sma + (std_dev * num_std_dev)

    def add_obv(self):
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['close'][i] > self.df['close'][i - 1]:
                obv.append(obv[-1] + self.df['volume'][i])
            elif self.df['close'][i] < self.df['close'][i - 1]:
                obv.append(obv[-1] - self.df['volume'][i])
            else:
                obv.append(obv[-1])
        self.df['obv'] = obv

    def add_vwap(self):
        cum_vol = self.df['volume'].cumsum()
        cum_vol_price = (self.df['close'] * self.df['volume']).cumsum()
        self.df['vwap'] = cum_vol_price / cum_vol
    
    def add_cci(self, period: int = 20):
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        self.df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)
    
    def add_mfi(self, period: int = 14):
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        money_flow = typical_price * self.df['volume']
        positive_flow = []
        negative_flow = []
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow.append(money_flow[i])
                negative_flow.append(0)
            elif typical_price[i] < typical_price[i - 1]:
                positive_flow.append(0)
                negative_flow.append(money_flow[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        mfr = positive_mf / negative_mf
        self.df[f'mfi_{period}'] = 100 - (100 / (1 + mfr))
    
    def add_stochastic_oscillator(self, k_period: int = 14, d_period: int = 3):
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()
        self.df['%K'] = 100 * ((self.df['close'] - low_min) / (high_max - low_min))
        self.df['%D'] = self.df['%K'].rolling(window=d_period).mean()
    
    def add_adx(self, period: int = 14):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame((high - close.shift()).abs())
        tr3 = pd.DataFrame((low - close.shift()).abs())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        self.df[f'adx_{period}'] = dx.rolling(window=period).mean()

    def add_fibonacci_retracement(self):
        max_price = self.df['high'].max()
        min_price = self.df['low'].min()
        diff = max_price - min_price
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in levels:
            self.df[f'fibonacci_{int(level*100)}'] = max_price - (diff * level)
        
    def add_ichimoku_cloud(self, conversion_period: int = 9, base_period: int = 26, span_b_period: int = 52, displacement: int = 26):
        high_9 = self.df['high'].rolling(window=conversion_period).max()
        low_9 = self.df['low'].rolling(window=conversion_period).min()
        self.df['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = self.df['high'].rolling(window=base_period).max()
        low_26 = self.df['low'].rolling(window=base_period).min()
        self.df['kijun_sen'] = (high_26 + low_26) / 2

        self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen']) / 2).shift(displacement)

        high_52 = self.df['high'].rolling(window=span_b_period).max()
        low_52 = self.df['low'].rolling(window=span_b_period).min()
        self.df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(displacement)

        self.df['chikou_span'] = self.df['close'].shift(-displacement)

    def add_col_by_str(self, indicator: str):
        indicator = indicator.upper()
        if "_" in indicator:
            func = indicator.split("_")[0]
            arg = int(indicator.split("_")[1])
            if func in self.col_by_str:
                self.col_by_str[func](arg)
            else:
                print(f"Indicator {indicator} not recognized.")
        else:
            if indicator in self.col_by_str:
                self.col_by_str[indicator]()
            else:
                print(f"Indicator {indicator} not recognized.")

    def add_col_by_list(self, indicators: list):
        for indicator in indicators:
            self.add_col_by_str(indicator)

if __name__ == "__main__":
    analyser = TechAnalyser()
    analyser.add_col_by_str("sma_5")
    print(analyser.df)





