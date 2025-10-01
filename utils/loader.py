import yaml
config_path = {
    "test_eod": "configs/fetch_history.yaml",
    "fetch_history": "configs/fetch_history.yaml",
    "simple_lstm": "configs/simple_lstm.yaml"
}
def load_config(args):
    print(f"Loading config for {args.task}...")
    with open(config_path[args.task], "r") as file:
        config_raw = yaml.safe_load(file)
    return config_raw

def load_config(model_name : str):
    print(f"Loading config for {model_name}...")
    with open(config_path[model_name], "r") as file:
        config_raw = yaml.safe_load(file)
    return config_raw

def load_dataset(path: str):
    print(f"Loading dataset from {path}...")
    import pandas as pd
    df = pd.read_csv(path)
    return df