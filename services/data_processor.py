from abc import ABC, abstractmethod
from utils.loader import load_config

class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, path: str): pass

    @abstractmethod
    def clean_data(self): pass

    @abstractmethod
    def transform_data(self): pass

class TimeSeriesProcessor(DataProcessor):
    def __init__(self, model_name : str):
        self.config = load_config(model_name)["dataset"]
        self.input_features = self.config["input_features"]
        self.output_features = self.config["output_features"]
        self.sequence_length = self.config["sequence_length"]
        self.timesteps = self.config["timesteps"]
        