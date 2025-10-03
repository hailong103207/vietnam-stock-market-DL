import pandas as pd
import yaml
def df_to_csv(df: pd.DataFrame, file_path: str):
    """Save DataFrame to CSV file."""
    df.to_csv(file_path, index=False)
    
def save_yaml(file_path : str, config):
    with open(file_path, "w"):
        yaml.dump(data=config, default_flow_style=False)