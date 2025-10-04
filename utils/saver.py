import pandas as pd
import yaml
def df_to_csv(df: pd.DataFrame, file_path: str):
    """Save DataFrame to CSV file."""
    df.to_csv(file_path, index=False)
    
def save_yaml(file_path : str, data):
    with open(file_path, "w") as yaml_file:
        print(data)
        print(file_path)
        dump = yaml.dump(data=data, default_flow_style=False)
        yaml_file.write(dump)

def save_cache(path_dir : str, data):
    if path_dir[-1] != "/":
        path_dir += "/"
    path = path_dir + "cache.yaml"
    save_yaml(path, data)
