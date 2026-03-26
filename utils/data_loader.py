"""
Data Loader Module
------------------
Handles loading and validating datasets for the Body Performance Analytics application.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os

# إعدادات التحقق من صحة البيانات
VALIDATION_RULES = {
    'age': {'min': 18, 'max': 100, 'dtype': 'int', 'name': 'Age (years)'},
    'gender': {'values': ['M', 'F', 'Male', 'Female'], 'dtype': 'category', 'name': 'Gender'},
    'height_cm': {'min': 100, 'max': 250, 'dtype': 'float', 'name': 'Height (cm)'},
    'weight_kg': {'min': 30, 'max': 200, 'dtype': 'float', 'name': 'Weight (kg)'},
    'body fat_%': {'min': 3, 'max': 50, 'dtype': 'float', 'name': 'Body Fat (%)'},
    'diastolic': {'min': 60, 'max': 120, 'dtype': 'float', 'name': 'Diastolic BP (mmHg)'},
    'systolic': {'min': 90, 'max': 200, 'dtype': 'float', 'name': 'Systolic BP (mmHg)'},
    'gripForce': {'min': 10, 'max': 100, 'dtype': 'float', 'name': 'Grip Force (kg)'},
    'sit_and_bend_forward_cm': {'min': -20, 'max': 50, 'dtype': 'float', 'name': 'Flexibility (cm)'},
    'sit-ups counts': {'min': 0, 'max': 100, 'dtype': 'int', 'name': 'Sit-ups Count'},
    'broad jump_cm': {'min': 50, 'max': 350, 'dtype': 'float', 'name': 'Broad Jump (cm)'},
    'class': {'values': ['A', 'B', 'C', 'D'], 'dtype': 'category', 'name': 'Performance Class'}
}

FEATURE_COLUMNS = [
    'age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
    'diastolic', 'systolic', 'gripForce', 'sit_and_bend_forward_cm', 'sit-ups counts'
]

TARGET_COLUMNS = ['class', 'broad jump_cm']


def load_data(
    file_path: str = 'data/bodyPerformance.csv',
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    sample_size : int, optional
        Number of rows to load (for testing)
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def validate_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate dataset against predefined rules.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    
    Returns:
    --------
    Tuple[bool, Dict]
        (is_valid, validation_report)
    """
    errors = []
    warnings = []
    
    # Check required columns
    required_cols = list(VALIDATION_RULES.keys()) + ['class']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Validate each column
    for col, rules in VALIDATION_RULES.items():
        if col not in df.columns:
            continue
        
        series = df[col]
        
        # Check data type
        if rules['dtype'] == 'int':
            if not pd.api.types.is_integer_dtype(series):
                warnings.append(f"{rules['name']} should be integer")
        
        # Check min/max
        if 'min' in rules and 'max' in rules:
            out_of_range = series[(series < rules['min']) | (series > rules['max'])].count()
            if out_of_range > 0:
                warnings.append(
                    f"{rules['name']}: {out_of_range} values outside range "
                    f"[{rules['min']}, {rules['max']}]"
                )
        
        # Check categorical values
        if 'values' in rules:
            invalid = series[~series.isin(rules['values'])].count()
            if invalid > 0:
                errors.append(
                    f"{rules['name']}: {invalid} invalid values. "
                    f"Allowed: {rules['values']}"
                )
    
    # Check class balance
    if 'class' in df.columns:
        class_counts = df['class'].value_counts()
        if len(class_counts) < 4:
            warnings.append(f"Missing classes: {set(['A','B','C','D']) - set(class_counts.index)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        warnings.append(f"Missing values found: {missing[missing > 0].to_dict()}")
    
    is_valid = len(errors) == 0
    
    report = {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'shape': df.shape,
        'columns': list(df.columns),
        'class_counts': df['class'].value_counts().to_dict() if 'class' in df.columns else {}
    }
    
    return is_valid, report


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to raw data (same as in training).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    df_clean = df.copy()
    
    # Cap outliers (same as in training)
    df_clean['sit_and_bend_forward_cm'] = df_clean['sit_and_bend_forward_cm'].clip(upper=42)
    df_clean['diastolic'] = df_clean['diastolic'].replace(0, 70)
    df_clean['broad jump_cm'] = df_clean['broad jump_cm'].replace(0, df_clean['broad jump_cm'].median())
    
    # Encode gender
    df_clean['gender'] = df_clean['gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
    
    # Encode class
    class_map = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
    df_clean['class_encoded'] = df_clean['class'].map(class_map)
    
    return df_clean


def get_feature_columns() -> list:
    """Return list of feature column names."""
    return FEATURE_COLUMNS


def get_target_columns() -> list:
    """Return list of target column names."""
    return TARGET_COLUMNS


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    
    Returns:
    --------
    Dict
        Dataset information
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict(),
        'class_distribution': df['class'].value_counts().to_dict() if 'class' in df.columns else {}
    }


if __name__ == "__main__":
    # Test the module
    print("Testing data_loader module...")
    
    try:
        df = load_data()
        print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        is_valid, report = validate_data(df)
        print(f"✅ Data valid: {is_valid}")
        
        if report['warnings']:
            print(f"⚠️ Warnings: {len(report['warnings'])}")
        
        print("\nClass distribution:")
        for cls, count in report['class_counts'].items():
            print(f"  {cls}: {count}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
