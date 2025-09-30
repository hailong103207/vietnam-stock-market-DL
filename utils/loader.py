import yaml
config_path = {
    "test_eod": "configs/fetch_history.yaml",
    "fetch_history": "configs/fetch_history.yaml",
    "simple_lstm": "configs/simple_lstm.yaml"
}
def load_config(args):
    with open(config_path[args.task], "r") as file:
        config_raw = yaml.safe_load(file)
    return config_raw

def load_config(model_name : str):
    with open(config_path[model_name], "r") as file:
        config_raw = yaml.safe_load(file)
    return config_raw