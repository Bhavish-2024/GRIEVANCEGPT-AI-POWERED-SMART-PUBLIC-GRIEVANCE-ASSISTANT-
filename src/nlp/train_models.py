"""
Model Training and Evaluation Pipeline for GrievanceGPT using CivicDex.
Trains reproducible baseline classifiers with TF-IDF and Logistic Regression
for Intent, Category, Department, and Urgency.
"""
import shutil
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_PATH = Path(__file__).resolve().parent.parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from src.config import (
    BASE_DIR,
    DATA_DIR,
    CIVICDEX_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    INTENT_MODEL_PATH,
    CATEGORY_MODEL_PATH,
    DEPARTMENT_MODEL_PATH,
    URGENCY_MODEL_PATH,
)
from src.nlp.preprocessing import preprocess_for_classification

def prepare_dataset() -> pd.DataFrame:
    """
    Ensure data/civicdex.csv exists and export train/validation/test splits.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check source file in workspace root
    source_adapted = BASE_DIR / "civicdex_adapted.csv"
    source_main = BASE_DIR / "civicdex_main.csv"
    
    if not CIVICDEX_CSV.exists():
        if source_adapted.exists():
            shutil.copy(source_adapted, CIVICDEX_CSV)
            print(f"[Dataset] Copied {source_adapted.name} to {CIVICDEX_CSV}")
        elif source_main.exists():
            shutil.copy(source_main, CIVICDEX_CSV)
            print(f"[Dataset] Copied {source_main.name} to {CIVICDEX_CSV}")
        else:
            raise FileNotFoundError(f"No CivicDex source file found in {BASE_DIR}")
            
    df = pd.read_csv(CIVICDEX_CSV)
    
    # Detailed Dataset Inspection
    print("\n" + "="*50)
    print("      CIVICDEX DATASET COMPREHENSIVE INSPECTION")
    print("="*50)
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Column Names: {list(df.columns)}")
    print(f"Missing Values per Column:\n{df.isnull().sum()}")
    print(f"Duplicate raw_text records: {df.duplicated(subset=['raw_text']).sum()}")
    print("\nLanguage Distribution:\n", df['language_type'].value_counts())
    print("\nIntent Distribution:\n", df['intent'].value_counts())
    print("\nCategory Distribution:\n", df['category'].value_counts())
    print("\nDepartment Distribution:\n", df['department'].value_counts())
    print("\nUrgency Distribution:\n", df['urgency'].value_counts())
    print("="*50 + "\n")
    
    # Save splits based on split column if present, or create reproducible splits
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
    else:
        from sklearn.model_selection import train_test_split
        train_df, rem_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(rem_df, test_size=0.5, random_state=42)
        
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    print(f"[Splits] Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
    return df

def build_vectorizer() -> FeatureUnion:
    """
    Build a hybrid character and word TF-IDF vectorizer.
    Excels on multilingual Tamil, Romanized Tanglish, and English text.
    """
    return FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            analyzer='word',
            sublinear_tf=True
        )),
        ('char_tfidf', TfidfVectorizer(
            ngram_range=(2, 5),
            min_df=2,
            analyzer='char_wb',
            sublinear_tf=True
        ))
    ])

def train_and_evaluate(
    target_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_path: Path
) -> Pipeline:
    """
    Train a Logistic Regression pipeline for a specific target column and evaluate performance.
    """
    print(f"\n---> Training Classifier for Target: [{target_name}]")
    
    # Prepare text with preprocessing
    X_train = train_df['raw_text'].fillna('').apply(preprocess_for_classification)
    y_train = train_df[target_name].astype(str)
    
    X_val = val_df['raw_text'].fillna('').apply(preprocess_for_classification)
    y_val = val_df[target_name].astype(str)
    
    X_test = test_df['raw_text'].fillna('').apply(preprocess_for_classification)
    y_test = test_df[target_name].astype(str)
    
    # Pipeline: FeatureUnion (Word+Char TF-IDF) -> Logistic Regression
    pipeline = Pipeline([
        ('features', build_vectorizer()),
        ('classifier', LogisticRegression(
            C=2.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Validation Evaluation
    val_preds = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
    print(f"Validation Accuracy: {val_acc:.4f} | Weighted F1: {val_f1:.4f}")
    
    # Test Evaluation
    test_preds = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)
    print(f"Test Accuracy:       {test_acc:.4f} | Weighted F1: {test_f1:.4f}")
    
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, test_preds, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Successfully saved {target_name} model to {model_path}")
    
    return pipeline

def main():
    df = prepare_dataset()
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Train separate models as specified
    train_and_evaluate("intent", train_df, val_df, test_df, INTENT_MODEL_PATH)
    train_and_evaluate("category", train_df, val_df, test_df, CATEGORY_MODEL_PATH)
    train_and_evaluate("department", train_df, val_df, test_df, DEPARTMENT_MODEL_PATH)
    train_and_evaluate("urgency", train_df, val_df, test_df, URGENCY_MODEL_PATH)
    
    print("\n[Complete] All 4 CivicDex classifiers trained and persisted successfully.")

if __name__ == "__main__":
    main()
