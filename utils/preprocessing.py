"""
Preprocessing Module
--------------------
Handles data preprocessing, encoding, and scaling for the Body Performance Analytics application.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


# Feature columns (must match training)
FEATURE_COLUMNS = [
    'age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
    'diastolic', 'systolic', 'gripForce', 'sit_and_bend_forward_cm', 'sit-ups counts'
]

# Column ranges for validation
VALIDATION_RANGES = {
    'age': (18, 100),
    'height_cm': (100, 250),
    'weight_kg': (30, 200),
    'body fat_%': (3, 50),
    'diastolic': (60, 120),
    'systolic': (90, 200),
    'gripForce': (10, 100),
    'sit_and_bend_forward_cm': (-20, 50),
    'sit-ups counts': (0, 100),
    'broad jump_cm': (50, 350),
}


def encode_gender(gender_value: str) -> int:
    """
    Encode gender string to numeric value.
    
    Parameters:
    -----------
    gender_value : str
        Gender string ('M', 'F', 'Male', 'Female')
    
    Returns:
    --------
    int
        1 for Male, 0 for Female
    """
    if isinstance(gender_value, (int, float)):
        return int(gender_value)
    
    gender_str = str(gender_value).strip().upper()
    
    if gender_str in ['M', 'MALE', '1']:
        return 1
    elif gender_str in ['F', 'FEMALE', '0']:
        return 0
    else:
        return 0  # Default to Female


def encode_class(class_value: str) -> int:
    """
    Encode class string to numeric value.
    
    Parameters:
    -----------
    class_value : str
        Class string ('A', 'B', 'C', 'D')
    
    Returns:
    --------
    int
        3 for A (best), 2 for B, 1 for C, 0 for D (worst)
    """
    class_map = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
    
    class_str = str(class_value).strip().upper()
    
    return class_map.get(class_str, 0)


def decode_class(encoded_class: int) -> str:
    """
    Decode numeric class to string label.
    
    Parameters:
    -----------
    encoded_class : int
        Numeric class (0, 1, 2, 3)
    
    Returns:
    --------
    str
        Class label ('D', 'C', 'B', 'A')
    """
    decode_map = {0: 'D', 1: 'C', 2: 'B', 3: 'A'}
    return decode_map.get(encoded_class, 'D')


def get_class_description(class_label: str) -> Dict[str, str]:
    """
    Get description and interpretation for a performance class.
    
    Parameters:
    -----------
    class_label : str
        Class label ('A', 'B', 'C', 'D')
    
    Returns:
    --------
    Dict
        Description and interpretation
    """
    descriptions = {
        'A': {
            'name': 'Excellent Performance',
            'description': 'Top 25% of participants. Exceptional fitness level.',
            'color': '#1e3a5f',
            'icon': '🏆',
            'recommendation': 'Maintain current training regimen.'
        },
        'B': {
            'name': 'Good Performance',
            'description': 'Above average fitness level. Strong performance.',
            'color': '#10b981',
            'icon': '💪',
            'recommendation': 'Continue consistent training. Small improvements possible.'
        },
        'C': {
            'name': 'Average Performance',
            'description': 'Average fitness level. Room for improvement.',
            'color': '#f59e0b',
            'icon': '📊',
            'recommendation': 'Focus on flexibility and endurance training.'
        },
        'D': {
            'name': 'Needs Improvement',
            'description': 'Below average fitness level. Significant improvement needed.',
            'color': '#ef4444',
            'icon': '⚠️',
            'recommendation': 'Consider structured fitness program focusing on core areas.'
        }
    }
    
    return descriptions.get(class_label, descriptions['D'])


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap outliers based on training data thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with capped outliers
    """
    df_clean = df.copy()
    
    # Cap flexibility (same as in training)
    if 'sit_and_bend_forward_cm' in df_clean.columns:
        df_clean['sit_and_bend_forward_cm'] = df_clean['sit_and_bend_forward_cm'].clip(upper=42)
    
    # Cap diastolic zeros
    if 'diastolic' in df_clean.columns:
        df_clean['diastolic'] = df_clean['diastolic'].replace(0, 70)
    
    # Cap broad jump zeros
    if 'broad jump_cm' in df_clean.columns:
        df_clean['broad jump_cm'] = df_clean['broad jump_cm'].replace(0, df_clean['broad jump_cm'].median())
    
    return df_clean


def validate_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data against expected ranges.
    
    Parameters:
    -----------
    data : dict
        Input data dictionary
    
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_warnings)
    """
    warnings = []
    
    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in data:
            value = data[col]
            if value is not None:
                try:
                    val = float(value)
                    if val < min_val or val > max_val:
                        warnings.append(f"{col}: {val} is outside recommended range [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    warnings.append(f"{col}: Invalid value '{value}'")
    
    return len(warnings) == 0, warnings


def preprocess_single_input(
    input_data: Dict[str, Any],
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess a single input sample.
    
    Parameters:
    -----------
    input_data : dict
        Raw input data
    scaler : StandardScaler, optional
        Fitted scaler (if None, returns unscaled features)
    
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (processed features, warnings)
    """
    warnings = []
    
    # Validate input
    is_valid, validation_warnings = validate_input(input_data)
    warnings.extend(validation_warnings)
    
    # Create feature array
    features = []
    
    for col in FEATURE_COLUMNS:
        if col in input_data:
            value = input_data[col]
            
            # Special handling for gender
            if col == 'gender':
                value = encode_gender(value)
            
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(0.0)
                warnings.append(f"{col}: Could not convert '{value}' to number, using 0")
        else:
            features.append(0.0)
            warnings.append(f"{col}: Missing value, using 0")
    
    X = np.array([features])
    
    # Scale if scaler provided
    if scaler is not None:
        X_scaled = scaler.transform(X)
        return X_scaled, warnings
    
    return X, warnings


def preprocess_batch_data(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    target_column: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Preprocess batch data for prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    scaler : StandardScaler, optional
        Fitted scaler
    target_column : str, optional
        Name of target column (if exists)
    
    Returns:
    --------
    Tuple[np.ndarray, Optional[np.ndarray], List[str]]
        (features, targets, warnings)
    """
    warnings = []
    df_clean = df.copy()
    
    # Cap outliers
    df_clean = cap_outliers(df_clean)
    
    # Encode gender
    if 'gender' in df_clean.columns:
        df_clean['gender'] = df_clean['gender'].apply(encode_gender)
    
    # Check for missing columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_clean.columns]
    if missing_cols:
        warnings.append(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            df_clean[col] = 0
    
    # Extract features
    X = df_clean[FEATURE_COLUMNS].values
    
    # Extract target if exists
    y = None
    if target_column and target_column in df_clean.columns:
        y = df_clean[target_column].values
    
    # Scale features
    if scaler is not None:
        X_scaled = scaler.transform(X)
        return X_scaled, y, warnings
    
    return X, y, warnings


def get_feature_names() -> List[str]:
    """
    Get list of feature column names.
    
    Returns:
    --------
    List[str]
        Feature names
    """
    return FEATURE_COLUMNS.copy()


def get_feature_importance_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about feature importance from EDA.
    
    Returns:
    --------
    Dict
        Feature importance information
    """
    return {
        'sit_and_bend_forward_cm': {
            'name': 'Flexibility',
            'correlation': 0.59,
            'importance': 'highest',
            'interpretation': 'Strongest predictor of performance class'
        },
        'sit-ups counts': {
            'name': 'Muscular Endurance',
            'correlation': 0.45,
            'importance': 'high',
            'interpretation': 'Second strongest predictor'
        },
        'body fat_%': {
            'name': 'Body Fat Percentage',
            'correlation': -0.34,
            'importance': 'high_negative',
            'interpretation': 'Strongest negative predictor'
        },
        'broad jump_cm': {
            'name': 'Explosive Power',
            'correlation': 0.26,
            'importance': 'moderate',
            'interpretation': 'Moderate positive correlation'
        },
        'gripForce': {
            'name': 'Grip Strength',
            'correlation': 0.22,
            'importance': 'moderate',
            'interpretation': 'Moderate positive correlation'
        },
        'age': {
            'name': 'Age',
            'correlation': -0.07,
            'importance': 'weak',
            'interpretation': 'Weak negative correlation'
        }
    }


if __name__ == "__main__":
    # Test the module
    print("Testing preprocessing module...")
    
    # Test encoding
    print(f"Gender 'M' -> {encode_gender('M')}")
    print(f"Gender 'F' -> {encode_gender('F')}")
    print(f"Class 'A' -> {encode_class('A')}")
    print(f"Class 'D' -> {encode_class('D')}")
    print(f"Decode 3 -> {decode_class(3)}")
    
    # Test class descriptions
    desc = get_class_description('A')
    print(f"Class A description: {desc['name']}")
    
    # Test feature names
    print(f"Features: {get_feature_names()}")
    
    print("\n✅ preprocessing.py loaded successfully")
