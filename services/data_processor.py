from abc import ABC, abstractmethod
from utils.loader import load_config
from utils.loader import load_dataset
import pandas as pd
import numpy as np
from .tech_analyser import TechAnalyser
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import os

class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, path: str): pass

    @abstractmethod
    def transform_data(self): pass

    @abstractmethod
    def fetch_dataset(self): pass

    @abstractmethod
    def get_dataset(self): pass

class TimeSeriesProcessor(DataProcessor, Dataset):
    def __init__(self, model_name : str):
        self.model_name_to_modify = {
            "simple_lstm": self.modify_dataset_simple_lstm
        }

        self.config = load_config(model_name)
        self.X_train, self.X_val, self.y_train, self.y_val = None, None, None, None
        self.cache_dir = self.config["cache_dir"]
        self.use_cache = self.config["cache"]["use_cache"]
        if self.use_cache:
            print("Using cached dataset...")
            self.X_train = torch.load(f"{self.cache_dir}/X_train.pt")
            self.y_train = torch.load(f"{self.cache_dir}/y_train.pt")
            self.X_val = torch.load(f"{self.cache_dir}/X_val.pt")
            self.y_val = torch.load(f"{self.cache_dir}/y_val.pt")
            print("Loaded dataset from cache.")
            print()
        else:
            self.config_dataset = self.config["dataset"]
            self.device = self.config["device"]
            self.max_row = self.config_dataset["max_row"]
            self.input_features = self.config_dataset["input_features"]
            self.output_features = self.config_dataset["output_features"]
            self.sequence_length = self.config_dataset["sequence_length"]
            self.timesteps = self.config_dataset["timesteps"]
            self.train_split = self.config_dataset["train_split"]
            self.dataset_path = self.config_dataset["dataset_path"]
            self.raw_df = self.load_data()
            self.tickers = self.raw_df['ticker'].unique()
            self.started_rows = None
            self.ended_rows = None
            self.modified_df = pd.DataFrame()
            self.transform_data()
            #Final dataset
            self.fetch_dataset()
            self.model_name_to_modify[model_name]()
            
            #Save to cache
            print("Saving dataset to cache...")
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(self.X_train, f"{self.cache_dir}/X_train.pt")
            torch.save(self.y_train, f"{self.cache_dir}/y_train.pt")
            torch.save(self.X_val, f"{self.cache_dir}/X_val.pt")
            torch.save(self.y_val, f"{self.cache_dir}/y_val.pt")
            print(f"Saved dataset to {self.cache_dir}")
            print()

    def time_series_generate(data:pd.DataFrame, sequence_length:int, timesteps:int, input_features:list, output_features:list):
        X, y = [], []
        for i in range(len(data) - sequence_length - timesteps + 1):
            X.append(data[input_features].iloc[i:i+sequence_length].values)
            y.append(data[output_features].iloc[i+sequence_length:i+sequence_length+timesteps-1].values)
        Xt = np.array(X)
        yt = np.array(y)
        return np.array(X), np.array(y)
        # for i in range(len(data) - timesteps - steps_to_predict + 1):
        #     X.append(data[i:i+timesteps].values)
        #     y.append(data[i+timesteps:i+timesteps+steps_to_predict].values)
        # return np.array(X), np.array(y)

    def load_data(self, path: str = None) -> pd.DataFrame:
        if path is None:
            path = self.dataset_path
        return load_dataset(path)

    def transform_data(self):
        print("Transforming data...")
        self.modified_df['ticker'] = self.raw_df['ticker']
        tech_analyser = TechAnalyser(self.raw_df)
        for feature in self.input_features:
            if feature not in self.raw_df.columns:
                tech_analyser.add_col_by_str(feature)
            self.modified_df[feature] = tech_analyser.df[feature]
        for feature in self.output_features:
            if feature not in self.raw_df.columns:
                tech_analyser.add_col_by_str(feature)
                self.modified_df[feature] = tech_analyser.df[feature]
        self.modified_df.dropna(inplace=True)
        self.modified_df.reset_index(drop=True, inplace=True)
        self.tickers = self.modified_df['ticker'].unique()
        self.started_rows = self.modified_df.groupby('ticker').head(1).index
        self.ended_rows = self.modified_df.groupby('ticker').tail(1).index
        print()
    def fetch_dataset(self):
        print("Preparing dataset...")
        # print(self.modified_df)
        X = []
        y = []
        cnt_rows = 0
        for i in range(len(self.tickers)):
            if(cnt_rows >= self.max_row):
                break
            df_ticker = pd.DataFrame()
            if i == len(self.tickers) - 1:
                df_ticker = self.modified_df.iloc[self.started_rows[i]:]
            else:
                df_ticker = self.modified_df.iloc[self.started_rows[i]:self.ended_rows[i] + 1]
            if(df_ticker.shape[0] < self.sequence_length + self.timesteps):
                continue
            print(f'Processing ticker {self.tickers[i]} with {df_ticker.shape[0]} rows.')
            predict_features_index = []
            X_temp, y_temp = TimeSeriesProcessor.time_series_generate(df_ticker, self.sequence_length, self.timesteps, self.input_features, self.output_features)
            X.extend(X_temp)
            y.extend(y_temp)
            cnt_rows += df_ticker.shape[0]
        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, train_size=self.train_split, shuffle=False, random_state=42)
        print()
    
    def modify_dataset_simple_lstm(self):
        #take last element of each sequence as label
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.y_train[:, -1], dtype=torch.float32).to(self.device)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(self.y_val[:, -1], dtype=torch.float32).to(self.device)
        # return self.X_train_tensor, self.y_train_tensor, self.X_val_tensor, self.y_val_tensor

    def get_dataset(self):
        return self.X_train, self.X_val, self.y_train, self.y_val

if __name__ == "__main__":
    processor = TimeSeriesProcessor("simple_lstm")
    X_train, X_test, y_train, y_test = processor.get_dataset()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")