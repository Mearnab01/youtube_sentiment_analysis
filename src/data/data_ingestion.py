from pathlib import Path
import yaml
import pandas as pd
from src.utils.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger("data_ingestion")

ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yaml"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

def load_params(config_path:str)->dict:
    try:
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    

def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file at {data_path}: {e}")
        raise
    
    
def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']
        logger.info("Data preprocessing completed successfully")
        return df
    except KeyError as e:
        logger.error(f"Missing expected column in DataFrame: {e}")
        raise
    
def save_data(train_df:pd.DataFrame, test_df:pd.DataFrame, data_path:str)-> None:
    try:
        raw_dir = data_path/"raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(raw_dir / "train.csv", index=False)
        test_df.to_csv(raw_dir / "test.csv", index=False)
        logger.info(f"Data saved successfully to {data_path}")
    except Exception as e:
        logger.error(f"Error saving data to {data_path}: {e}")
        raise
    
def main():
    try:
        logger.info(">> Data Ingestion Phase...")
        params = load_params(PARAMS_PATH)

        test_size = params["data_ingestion"]["test_size"]

        df = load_data(
            "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        )

        df = preprocess_data(df)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
        )

        save_data(train_df, test_df, DATA_DIR)

        logger.info("Data ingestion completed successfully.")

    except Exception:
        logger.exception("Data ingestion pipeline failed.")
        raise

if __name__ == "__main__":
    main()