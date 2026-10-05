"""
Unit tests for CivicDex dataset loading, integrity, and splits.
"""
import pandas as pd
from src.config import CIVICDEX_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV

def test_dataset_exists_and_valid():
    assert CIVICDEX_CSV.exists(), "civicdex.csv must exist in data/"
    df = pd.read_csv(CIVICDEX_CSV)
    assert len(df) >= 600, "CivicDex dataset should have at least 600 rows"
    
    required_cols = [
        'request_id', 'raw_text', 'normalized_text', 'english_gloss',
        'language_type', 'intent', 'category', 'department', 'urgency', 'severity'
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column {col} missing from dataset"

def test_splits_integrity():
    assert TRAIN_CSV.exists()
    assert VAL_CSV.exists()
    assert TEST_CSV.exists()
    
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
