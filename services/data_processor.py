from abc import ABC, abstractmethod
from utils.loader import load_config
from utils.loader import load_dataset
import pandas as pd
import numpy as np
from tech_analyser import TechAnalyser

class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, path: str): pass

    @abstractmethod
    def transform_data(self): pass

class TimeSeriesProcessor(DataProcessor):
    def __init__(self, model_name : str):
        self.config = load_config(model_name)["dataset"]
        self.input_features = self.config["input_features"]
        self.output_features = self.config["output_features"]
        self.sequence_length = self.config["sequence_length"]
        self.timesteps = self.config["timesteps"]
        self.train_split = self.config["train_split"]
        self.dataset_path = self.config["dataset_path"]
        self.raw_df = self.load_data(self.dataset_path)
        self.modified_df = pd.DataFrame()
        self.X = None
        self.y = None
        self.transform_data()

    def load_data(self, path: str = None) -> pd.DataFrame:
        if path is None:
            path = self.dataset_path
        return load_dataset(path)

    def transform_data(self):
        tech_analyser = TechAnalyser(self.raw_df)
        tech_analyser.add_col_by_list(self.input_features)
        for feature in self.output_features:
            self.modified_df.add_col_by_list(feature)
        

