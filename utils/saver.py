import pandas as pd

def df_to_csv(df: pd.DataFrame, file_path: str):
    """Save DataFrame to CSV file."""
    df.to_csv(file_path, index=False)
    