import yaml
def load_config(args):
    config_path = {
        "test_eod": "configs/fetch_eod.yaml",
        "fetch_eod": "configs/fetch_eod.yaml",
    }
    with open(config_path[args.task], "r") as file:
        config_raw = yaml.safe_load(file)
    return config_raw