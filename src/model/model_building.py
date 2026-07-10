import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import get_logger

logger = get_logger("model_building")

ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yaml"
DATA_DIR = ROOT_DIR / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"

def load_params(config_path: str) -> dict:
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
    
def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        df.fillna("", inplace=True)
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file at {data_path}: {e}")
        raise
    
def apply_tfidf(train_data:pd.DataFrame, max_features:int, ngram_range:tuple)->tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.info(f"TF-IDF vectorization applied successfully with max_features={max_features} and ngram_range={ngram_range}")
        
        vectorizer_path = get_root_dir() / 'models' / 'tfidf_vectorizer.pkl'
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
            
        logger.info("TF-IDF vectorizer saved successfully as 'tfidf_vectorizer.pkl'")
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise
    
def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
) -> lgb.LGBMClassifier:
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X_train, y_train)
        logger.info(f"LightGBM model trained successfully with learning_rate={learning_rate}, max_depth={max_depth}, n_estimators={n_estimators}")
        return model
    except Exception as e:
        logger.error(f"Error training LightGBM model: {e}")
        raise
    
def save_model(model: lgb.LGBMClassifier, model_path: str) -> None:
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")
        raise
    
def get_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]

def main():
    try:
        logger.info(">>> Starting model building process")
        params = load_params(PARAMS_PATH)
        train_data_path = INTERIM_DATA_DIR / "train_processed.csv"
        train_data = load_data(train_data_path)
        
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)
        
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']
        
        model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)
        
        model_path = get_root_dir() / "models" / "lgbm_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, model_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()