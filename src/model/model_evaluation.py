import json
import pickle
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
 
MODEL_PATH = MODELS_DIR / "lgbm_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
TEST_DATA_PATH = INTERIM_DIR / "test_processed.csv"
 
REPORT_PATH = ARTIFACTS_DIR / "classification_report.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
EXPERIMENT_INFO_PATH = ARTIFACTS_DIR / "experiment_info.json"

logger = get_logger("model_evaluation")

def load_data(file_path: Path)-> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        raise
    
def load_model(model_path: Path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    
def load_vectorizer(vectorizer_path: Path) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise
    
def load_params(params_path: Path) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise
    
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise
 
 
def save_classification_report(report: dict, file_path: Path) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(report, file, indent=4)
        logger.debug('Classification report saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the classification report: %s', e)
        raise
 
 
def save_confusion_matrix(cm, dataset_name: str, file_path: Path) -> None:
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(file_path)
        plt.close()
        logger.debug('Confusion matrix saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the confusion matrix: %s', e)
        raise
 
 
def save_experiment_info(params: dict, report: dict, file_path: Path) -> None:
    try:
        experiment_info = {
            'params': params,
            'accuracy': report.get('accuracy'),
            'macro_avg': report.get('macro avg'),
            'weighted_avg': report.get('weighted avg'),
        }
        with open(file_path, 'w') as file:
            json.dump(experiment_info, file, indent=4)
        logger.debug('Experiment info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the experiment info: %s', e)
        raise
 
 
# ==========================================================
# Main
# ==========================================================
def main():
    try:
        logger.info(">>> Starting model evaluation process")
        params = load_params(ROOT_DIR / 'params.yaml')
 
        model = load_model(MODEL_PATH)
        vectorizer = load_vectorizer(VECTORIZER_PATH)
 
        test_data = load_data(TEST_DATA_PATH)
 
        X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
        y_test = test_data['category'].values
 
        report, cm = evaluate_model(model, X_test_tfidf, y_test)
 
        save_classification_report(report, REPORT_PATH)
        save_confusion_matrix(cm, "Test Data", CONFUSION_MATRIX_PATH)
        save_experiment_info(params, report, EXPERIMENT_INFO_PATH)
 
        logger.debug('Model evaluation pipeline completed successfully')
 
    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        print(f"Error: {e}")
 
 
if __name__ == '__main__':
    main()
 