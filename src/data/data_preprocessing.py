import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from src.utils.logger import get_logger

logger = get_logger("data_preprocessing")

ROOT_DIR = Path(__file__).resolve().parents[2]

# Paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"


# download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')



def preprocess_comment(text):
    try:
        comment = text.lower()
        comment = comment.strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        
        return comment
        
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return None
    
def normalize_text(df):
    try:
        df = df.copy()
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.info("Preprocess Completed..")
        return df
    except Exception as e:
        logger.error(f"Error in normalizing text: {e}")
        return None

def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str)->None:
    try:
        interim_dir = data_path / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(
            interim_dir / "train_processed.csv",
            index=False,
        )

        test_data.to_csv(
            interim_dir / "test_processed.csv",
            index=False,
        )

        logger.info(f"Processed datasets saved to {interim_dir}")
    except Exception as e:
        logger.error(f"Error in saving data: {e}")
        return None
    
def main():
    try:
        logger.info(">> Data Preprocessing Phase...")
        
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("data loaded successfully...")
        
        train_preprocessed_data = normalize_text(train_data)
        test_preprocessed_data = normalize_text(test_data)
        
        save_data(
            train_preprocessed_data,
            test_preprocessed_data,
            DATA_DIR
        )
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return None
    
if __name__ == "__main__":
    main()