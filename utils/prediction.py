"""
Prediction Module
-----------------
Handles predictions with trained machine learning models for the Body Performance Analytics application.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Union
import streamlit as st


def predict_classification(
    model: Any,
    features_scaled: np.ndarray,
    return_proba: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Make classification prediction.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained classification model
    features_scaled : np.ndarray
        Scaled input features (shape: n_samples, n_features)
    return_proba : bool
        Whether to return probabilities (default: True)
    
    Returns:
    --------
    If return_proba:
        Tuple[predicted_classes, predicted_probabilities]
    Else:
        predicted_classes
    """
    try:
        # Make prediction
        predictions = model.predict(features_scaled)
        
        # Map numeric predictions to class labels
        class_map = {0: 'D', 1: 'C', 2: 'B', 3: 'A'}
        class_labels = np.array([class_map[p] for p in predictions])
        
        if return_proba and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)
            return class_labels, probabilities
        elif return_proba:
            # If model doesn't have predict_proba (like SVM without probability=True)
            # Return confidence scores or None
            return class_labels, None
        
        return class_labels
    
    except Exception as e:
        raise Exception(f"Error in classification prediction: {str(e)}")


def predict_regression(
    model: Any,
    features_scaled: np.ndarray
) -> np.ndarray:
    """
    Make regression prediction.
    
    Parameters:
    -----------
    model : sklearn regressor
        Trained regression model
    features_scaled : np.ndarray
        Scaled input features (shape: n_samples, n_features)
    
    Returns:
    --------
    np.ndarray
        Predicted values
    """
    try:
        predictions = model.predict(features_scaled)
        return predictions
    except Exception as e:
        raise Exception(f"Error in regression prediction: {str(e)}")


def predict_batch(
    models: Dict[str, Any],
    data: pd.DataFrame,
    scaler: Any,
    model_type: str = 'classification',
    model_name: str = None
) -> pd.DataFrame:
    """
    Make batch predictions on multiple samples.
    
    Parameters:
    -----------
    models : dict
        Dictionary of loaded models
    data : pd.DataFrame
        Input data (raw, unscaled)
    scaler : sklearn scaler
        Fitted StandardScaler
    model_type : str
        'classification' or 'regression'
    model_name : str
        Specific model name to use (if None, use best model)
    
    Returns:
    --------
    pd.DataFrame
        Original data with predictions added
    """
    try:
        # Prepare features
        feature_cols = [
            'age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
            'diastolic', 'systolic', 'gripForce', 'sit_and_bend_forward_cm', 'sit-ups counts'
        ]
        
        # Check if all features exist
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Encode gender if needed
        df = data.copy()
        if df['gender'].dtype == 'object':
            df['gender'] = df['gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
        
        # Select features
        X = df[feature_cols].values
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Select model
        if model_name is None:
            if model_type == 'classification':
                model_name = 'mlp'  # Best classifier
            else:
                model_name = 'mlp_regressor'  # Best regressor
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = models[model_name]
        
        # Make predictions
        if model_type == 'classification':
            predictions, probabilities = predict_classification(model, X_scaled, return_proba=True)
            
            # Add predictions to dataframe
            df['predicted_class'] = predictions
            
            if probabilities is not None:
                class_names = ['D', 'C', 'B', 'A']
                for i, class_name in enumerate(class_names):
                    df[f'prob_{class_name}'] = probabilities[:, i]
        else:
            predictions = predict_regression(model, X_scaled)
            df['predicted_broad_jump_cm'] = predictions
            df['predicted_broad_jump_cm'] = df['predicted_broad_jump_cm'].round(1)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error in batch prediction: {str(e)}")


def predict_with_confidence(
    model: Any,
    features_scaled: np.ndarray,
    class_names: List[str] = ['D', 'C', 'B', 'A']
) -> Dict[str, Any]:
    """
    Make prediction with confidence level and interpretation.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained classification model
    features_scaled : np.ndarray
        Scaled input features (single sample)
    class_names : list
        Names of classes in order
    
    Returns:
    --------
    Dict
        Prediction result with class, confidence, and interpretation
    """
    try:
        # Get prediction
        pred_class, proba = predict_classification(model, features_scaled, return_proba=True)
        
        # Get confidence
        confidence = np.max(proba) if proba is not None else None
        
        # Interpretation based on confidence
        if confidence is not None:
            if confidence >= 0.8:
                interpretation = "High confidence prediction"
                recommendation = "The model is very certain about this classification."
            elif confidence >= 0.6:
                interpretation = "Moderate confidence prediction"
                recommendation = "The model is reasonably confident. Consider additional validation."
            elif confidence >= 0.4:
                interpretation = "Low confidence prediction"
                recommendation = "The model is uncertain. Review input data or use ensemble methods."
            else:
                interpretation = "Very low confidence prediction"
                recommendation = "The model is highly uncertain. Input may be outside training distribution."
        else:
            interpretation = "Confidence not available for this model"
            recommendation = "Consider using a model with probability estimates."
        
        return {
            'predicted_class': pred_class[0],
            'probabilities': proba[0] if proba is not None else None,
            'confidence': confidence,
            'interpretation': interpretation,
            'recommendation': recommendation
        }
    
    except Exception as e:
        raise Exception(f"Error in prediction with confidence: {str(e)}")


def compare_models_predictions(
    models: Dict[str, Any],
    features_scaled: np.ndarray,
    model_type: str = 'classification'
) -> pd.DataFrame:
    """
    Get predictions from all models for comparison.
    
    Parameters:
    -----------
    models : dict
        Dictionary of loaded models
    features_scaled : np.ndarray
        Scaled input features
    model_type : str
        'classification' or 'regression'
    
    Returns:
    --------
    pd.DataFrame
        Predictions from all models
    """
    results = {}
    
    if model_type == 'classification':
        model_keys = ['knn', 'dt', 'svm_linear', 'svm_rbf', 'mlp']
        display_names = {
            'knn': 'KNN',
            'dt': 'Decision Tree',
            'svm_linear': 'SVM-Linear',
            'svm_rbf': 'SVM-RBF',
            'mlp': 'Neural Network'
        }
        
        for key in model_keys:
            if key in models and models[key] is not None:
                try:
                    pred_class, _ = predict_classification(models[key], features_scaled, return_proba=False)
                    results[display_names[key]] = pred_class[0]
                except Exception as e:
                    results[display_names[key]] = f"Error: {str(e)}"
            else:
                results[display_names[key]] = "Model not available"
    
    else:
        model_keys = ['linear_regression', 'dt_regressor', 'svr', 'mlp_regressor']
        display_names = {
            'linear_regression': 'Linear Regression',
            'dt_regressor': 'Decision Tree Regressor',
            'svr': 'SVR',
            'mlp_regressor': 'MLP Regressor'
        }
        
        for key in model_keys:
            if key in models and models[key] is not None:
                try:
                    pred = predict_regression(models[key], features_scaled)
                    results[display_names[key]] = round(pred[0], 1)
                except Exception as e:
                    results[display_names[key]] = f"Error: {str(e)}"
            else:
                results[display_names[key]] = "Model not available"
    
    return pd.DataFrame([results])


def get_prediction_summary(
    prediction_result: Dict[str, Any]
) -> str:
    """
    Generate a text summary of the prediction result.
    
    Parameters:
    -----------
    prediction_result : dict
        Output from predict_with_confidence
    
    Returns:
    --------
    str
        Human-readable summary
    """
    if prediction_result['confidence'] is not None:
        summary = f"""
        **Prediction:** Class {prediction_result['predicted_class']}
        **Confidence:** {prediction_result['confidence']:.1%}
        **Interpretation:** {prediction_result['interpretation']}
        
        **Recommendation:** {prediction_result['recommendation']}
        """
    else:
        summary = f"""
        **Prediction:** Class {prediction_result['predicted_class']}
        **Interpretation:** {prediction_result['interpretation']}
        **Recommendation:** {prediction_result['recommendation']}
        """
    
    return summary


if __name__ == "__main__":
    # Test the module
    print("Testing prediction module...")
    print("✅ prediction.py loaded successfully")
