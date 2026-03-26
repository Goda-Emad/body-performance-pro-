"""
Utils Package for Body Performance Analytics
--------------------------------------------
This package contains helper functions for:
- data_loader: Load and validate data
- model_loader: Load trained models
- preprocessing: Scale and encode input data
- prediction: Make predictions with trained models
- report_generator: Generate PDF reports
- visualizations: Create interactive plots
"""

from .data_loader import load_data, validate_data
from .model_loader import load_models, load_single_model
from .preprocessing import preprocess_input, preprocess_batch
from .prediction import predict_classification, predict_regression, predict_batch
from .report_generator import generate_pdf_report, generate_summary
from .visualizations import create_confusion_matrix, create_comparison_chart

__version__ = "1.0.0"
__author__ = "Team Body Performance Analytics"

__all__ = [
    # Data
    'load_data',
    'validate_data',
    # Models
    'load_models',
    'load_single_model',
    # Preprocessing
    'preprocess_input',
    'preprocess_batch',
    # Prediction
    'predict_classification',
    'predict_regression',
    'predict_batch',
    # Report
    'generate_pdf_report',
    'generate_summary',
    # Visualization
    'create_confusion_matrix',
    'create_comparison_chart',
]
