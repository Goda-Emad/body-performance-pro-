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

__version__ = "1.0.0"
__author__ = "Team Body Performance Analytics"

# ============================================
# Data Loader
# ============================================
try:
    from .data_loader import (
        load_data,
        validate_data,
        get_feature_columns,
        get_target_columns,
        get_dataset_info,
        preprocess_raw_data
    )
except ImportError:
    # Fallback if file doesn't exist
    load_data = None
    validate_data = None
    get_feature_columns = None
    get_target_columns = None
    get_dataset_info = None
    preprocess_raw_data = None

# ============================================
# Model Loader
# ============================================
try:
    from .model_loader import (
        load_models,
        load_single_model,
        get_available_models,
        get_model_info,
        check_models_integrity
    )
except ImportError:
    # Fallback if file doesn't exist
    load_models = None
    load_single_model = None
    get_available_models = None
    get_model_info = None
    check_models_integrity = None

# ============================================
# Preprocessing
# ============================================
try:
    from .preprocessing import (
        encode_gender,
        encode_class,
        decode_class,
        get_class_description,
        cap_outliers,
        validate_input,
        preprocess_single_input,
        preprocess_batch_data,
        get_feature_names,
        get_feature_importance_info
    )
except ImportError:
    # Fallback if file doesn't exist
    encode_gender = None
    encode_class = None
    decode_class = None
    get_class_description = None
    cap_outliers = None
    validate_input = None
    preprocess_single_input = None
    preprocess_batch_data = None
    get_feature_names = None
    get_feature_importance_info = None

# ============================================
# Prediction
# ============================================
try:
    from .prediction import (
        predict_classification,
        predict_regression,
        predict_batch,
        predict_with_confidence,
        compare_models_predictions,
        get_prediction_summary
    )
except ImportError:
    # Fallback if file doesn't exist
    predict_classification = None
    predict_regression = None
    predict_batch = None
    predict_with_confidence = None
    compare_models_predictions = None
    get_prediction_summary = None

# ============================================
# Report Generator
# ============================================
try:
    from .report_generator import (
        ReportGenerator,
        generate_summary
    )
except ImportError:
    # Fallback if file doesn't exist
    ReportGenerator = None
    generate_summary = None

# ============================================
# Visualizations
# ============================================
try:
    from .visualizations import (
        create_confusion_matrix,
        create_comparison_chart,
        create_model_comparison_dashboard,
        create_feature_importance_chart,
        create_prediction_gauge,
        create_distribution_plot,
        create_correlation_heatmap,
        create_scatter_colored,
        fig_to_base64,
        create_static_chart
    )
except ImportError:
    # Fallback if file doesn't exist
    create_confusion_matrix = None
    create_comparison_chart = None
    create_model_comparison_dashboard = None
    create_feature_importance_chart = None
    create_prediction_gauge = None
    create_distribution_plot = None
    create_correlation_heatmap = None
    create_scatter_colored = None
    fig_to_base64 = None
    create_static_chart = None

# ============================================
# Exports
# ============================================
__all__ = [
    # Data
    'load_data',
    'validate_data',
    'get_feature_columns',
    'get_target_columns',
    'get_dataset_info',
    'preprocess_raw_data',
    # Models
    'load_models',
    'load_single_model',
    'get_available_models',
    'get_model_info',
    'check_models_integrity',
    # Preprocessing
    'encode_gender',
    'encode_class',
    'decode_class',
    'get_class_description',
    'cap_outliers',
    'validate_input',
    'preprocess_single_input',
    'preprocess_batch_data',
    'get_feature_names',
    'get_feature_importance_info',
    # Prediction
    'predict_classification',
    'predict_regression',
    'predict_batch',
    'predict_with_confidence',
    'compare_models_predictions',
    'get_prediction_summary',
    # Report
    'ReportGenerator',
    'generate_summary',
    # Visualization
    'create_confusion_matrix',
    'create_comparison_chart',
    'create_model_comparison_dashboard',
    'create_feature_importance_chart',
    'create_prediction_gauge',
    'create_distribution_plot',
    'create_correlation_heatmap',
    'create_scatter_colored',
    'fig_to_base64',
    'create_static_chart',
]
