"""
Page: Generate Report - Simplified Version
-------------------------------------------
Generate summary reports without PDF (uses CSV/Excel download instead).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import io
import time

# إعدادات الصفحة
st.set_page_config(
    page_title="Generate Report | Body Performance Analytics",
    page_icon="📄",
    layout="wide"
)

# تحميل CSS
def load_css():
    try:
        with open('assets/style.css', 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except:
        pass

load_css()

# العنوان
st.markdown("""
<div class="main-header">
    <div class="main-title">📄 Generate Report</div>
    <div class="main-subtitle">Create comprehensive reports with analysis and predictions</div>
</div>
""", unsafe_allow_html=True)

# بيانات النماذج (من نتائج التدريب)
CLASSIFICATION_RESULTS = {
    'KNN (k=9)': {'Accuracy': 0.6316, 'Precision': 0.6510, 'Recall': 0.6316, 'F1 Score': 0.6352},
    'Decision Tree': {'Accuracy': 0.6745, 'Precision': 0.6945, 'Recall': 0.6745, 'F1 Score': 0.6746},
    'SVM-Linear': {'Accuracy': 0.6114, 'Precision': 0.6113, 'Recall': 0.6114, 'F1 Score': 0.6103},
    'SVM-RBF': {'Accuracy': 0.6887, 'Precision': 0.6968, 'Recall': 0.6887, 'F1 Score': 0.6904},
    'Neural Network (MLP)': {'Accuracy': 0.7215, 'Precision': 0.7300, 'Recall': 0.7215, 'F1 Score': 0.7231}
}

REGRESSION_RESULTS = {
    'Linear Regression': {'R²': 0.7658, 'RMSE': 19.28, 'MAE': 15.12, 'MSE': 371.8},
    'Decision Tree Regressor': {'R²': 0.7221, 'RMSE': 21.00, 'MAE': 16.45, 'MSE': 441.0},
    'SVR': {'R²': 0.7749, 'RMSE': 18.90, 'MAE': 14.89, 'MSE': 357.2},
    'MLP Regressor': {'R²': 0.7791, 'RMSE': 18.73, 'MAE': 14.72, 'MSE': 350.8}
}

# تقسيم الصفحة
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Report Configuration</div>', unsafe_allow_html=True)
    
    # Report type
    report_type = st.radio(
        "Report Type",
        ["Full Analysis Report", "Single Prediction Report", "Batch Summary Report"],
        help="Choose the type of report you want to generate"
    )
    
    st.markdown("---")
    
    if report_type == "Single Prediction Report":
        st.markdown("### 👤 Participant Data")
        
        # Input form for single prediction
        col_a, col_b = st.columns(2)
        
        with col_a:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            body_fat = st.number_input("Body Fat (%)", min_value=3.0, max_value=50.0, value=20.0)
        
        with col_b:
            diastolic = st.number_input("Diastolic BP", min_value=60, max_value=120, value=80)
            systolic = st.number_input("Systolic BP", min_value=90, max_value=200, value=120)
            grip_force = st.number_input("Grip Force (kg)", min_value=10, max_value=100, value=40)
            flexibility = st.number_input("Flexibility (cm)", min_value=-20, max_value=50, value=15)
            sit_ups = st.number_input("Sit-ups Count", min_value=0, max_value=100, value=40)
        
        input_data = {
            'age': age,
            'gender': gender,
            'height_cm': height,
            'weight_kg': weight,
            'body fat_%': body_fat,
            'diastolic': diastolic,
            'systolic': systolic,
            'gripForce': grip_force,
            'sit_and_bend_forward_cm': flexibility,
            'sit-ups counts': sit_ups
        }
        
        # Manual prediction input
        pred_class = st.selectbox("Predicted Class (for report)", ["A (Best)", "B", "C", "D (Worst)"])
        pred_jump = st.number_input("Predicted Broad Jump (cm)", min_value=50.0, max_value=350.0, value=190.0)
        
        predictions = {
            'classification': {
                'predicted_class': pred_class[0],
                'confidence': 0.72
            },
            'regression': {
                'predicted_value': pred_jump
            }
        }
        
    elif report_type == "Batch Summary Report":
        st.markdown("### 📁 Batch Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with predictions",
            type=['csv', 'xlsx'],
            help="Upload a file with prediction results"
        )
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(batch_df)} rows")
            st.dataframe(batch_df.head(5), use_container_width=True)
            
            predictions = None
            input_data = {}
    
    else:  # Full Analysis Report
        st.markdown("### 📊 Report Options")
        
        include_classification = st.checkbox("Include Classification Results", value=True)
        include_regression = st.checkbox("Include Regression Results", value=True)
        include_cv = st.checkbox("Include Cross Validation", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        predictions = None
        input_data = {}
    
    st.markdown("---")
    
    # Additional options
    st.markdown("### 🎨 Formatting Options")
    
    col_a, col_b = st.columns(2)
    with col_a:
        include_charts = st.checkbox("Include Charts", value=True)
    with col_b:
        include_tables = st.checkbox("Include Tables", value=True)
    
    st.markdown("---")
    
    # Generate button
    generate_btn = st.button("📊 Generate Report", type="primary", use_container_width=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📄 Report Preview</div>', unsafe_allow_html=True)
    
    # Preview content
    st.markdown("### 📋 Report Summary")
    
    st.markdown(f"""
    **Report Type:** {report_type}
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    if report_type == "Single Prediction Report" and 'input_data' in locals():
        st.markdown("#### Participant Data")
        preview_df = pd.DataFrame([input_data]).T.reset_index()
        preview_df.columns = ['Feature', 'Value']
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Prediction Results")
        st.markdown(f"- **Predicted Class:** {pred_class}")
        st.markdown(f"- **Predicted Broad Jump:** {pred_jump:.1f} cm")
    
    elif report_type == "Batch Summary Report" and 'batch_df' in locals():
        st.markdown("#### Batch Summary")
        st.metric("Total Records", len(batch_df))
        if 'predicted_class' in batch_df.columns:
            st.metric("Classes", batch_df['predicted_class'].nunique())
        if 'predicted_broad_jump_cm' in batch_df.columns:
            st.metric("Avg Jump", f"{batch_df['predicted_broad_jump_cm'].mean():.1f} cm")
    
    else:
        st.markdown("#### Model Performance Summary")
        best_clf = max(CLASSIFICATION_RESULTS.keys(), key=lambda x: CLASSIFICATION_RESULTS[x]['Accuracy'])
        best_reg = max(REGRESSION_RESULTS.keys(), key=lambda x: REGRESSION_RESULTS[x]['R²'])
        
        st.markdown(f"- **Best Classifier:** {best_clf} ({CLASSIFICATION_RESULTS[best_clf]['Accuracy']:.1%})")
        st.markdown(f"- **Best Regressor:** {best_reg} (R² = {REGRESSION_RESULTS[best_reg]['R²']:.3f})")
        st.markdown(f"- **Key Predictor:** Flexibility (r = +0.59)")
    
    st.markdown("---")
    st.markdown("#### Key Insights")
    insights = [
        "MLP Neural Network achieves 72.15% accuracy",
        "MLP Regressor explains 77.9% of variance",
        "Flexibility is the strongest predictor",
        "10-Fold CV confirms model stability"
    ]
    for insight in insights:
        st.markdown(f"- {insight}")

st.markdown('</div>', unsafe_allow_html=True)

# Generate Report
if generate_btn:
    with st.spinner("🔄 Generating report..."):
        try:
            # Create results dataframe
            results_data = []
            
            if report_type == "Full Analysis Report" or report_type == "Single Prediction Report":
                # Add classification results
                for model, metrics in CLASSIFICATION_RESULTS.items():
                    row = {
                        'Model': model,
                        'Type': 'Classification',
                        'Accuracy': f"{metrics['Accuracy']:.1%}",
                        'Precision': f"{metrics['Precision']:.1%}",
                        'Recall': f"{metrics['Recall']:.1%}",
                        'F1 Score': f"{metrics['F1 Score']:.1%}",
                        'RMSE': '-',
                        'R²': '-'
                    }
                    results_data.append(row)
                
                # Add regression results
                for model, metrics in REGRESSION_RESULTS.items():
                    row = {
                        'Model': model,
                        'Type': 'Regression',
                        'Accuracy': '-',
                        'Precision': '-',
                        'Recall': '-',
                        'F1 Score': '-',
                        'RMSE': f"{metrics['RMSE']:.1f}",
                        'R²': f"{metrics['R²']:.3f}"
                    }
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                
                # Create summary dataframe
                summary_data = {
                    'Metric': ['Best Classifier', 'Best Regressor', 'Key Predictor', 'CV Stability'],
                    'Value': [
                        f"MLP Neural Network ({CLASSIFICATION_RESULTS['Neural Network (MLP)']['Accuracy']:.1%})",
                        f"MLP Regressor (R² = {REGRESSION_RESULTS['MLP Regressor']['R²']:.3f})",
                        "Flexibility (r = +0.59)",
                        "MLP: 72.98% ± 1.39%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                
            elif report_type == "Single Prediction Report" and 'input_data' in locals():
                # Single prediction report
                results_df = pd.DataFrame([input_data]).T
                results_df.columns = ['Value']
                results_df.index.name = 'Feature'
                results_df = results_df.reset_index()
                
                summary_df = pd.DataFrame({
                    'Metric': ['Predicted Class', 'Predicted Broad Jump'],
                    'Value': [pred_class, f"{pred_jump:.1f} cm"]
                })
                
            else:
                results_df = pd.DataFrame({'Message': ['No data available']})
                summary_df = pd.DataFrame({'Message': ['Upload a file to generate report']})
            
            # CSV download
            csv_output = results_df.to_csv(index=False)
            b64_csv = base64.b64encode(csv_output.encode()).decode()
            
            # Summary CSV
            summary_csv = summary_df.to_csv(index=False)
            b64_summary = base64.b64encode(summary_csv.encode()).decode()
            
            st.success("✅ Report generated successfully!")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<a href="data:file/csv;base64,{b64_csv}" download="performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" style="display: inline-block; padding: 0.5rem 1rem; background-color: #1e3a5f; color: white; border-radius: 8px; text-decoration: none; text-align: center; width: 100%;">📥 Download Full Report (CSV)</a>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f'<a href="data:file/csv;base64,{b64_summary}" download="summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" style="display: inline-block; padding: 0.5rem 1rem; background-color: #2c4e6e; color: white; border-radius: 8px; text-decoration: none; text-align: center; width: 100%;">📥 Download Summary (CSV)</a>', unsafe_allow_html=True)
            
            # Display preview
            st.markdown("---")
            st.markdown("### 📊 Report Preview")
            st.dataframe(results_df.head(20), use_container_width=True)
            
            if len(results_df) > 20:
                st.caption(f"Showing first 20 of {len(results_df)} rows")
            
        except Exception as e:
            st.error(f"❌ Error generating report: {e}")
            st.info("Please check your inputs and try again.")

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ Report Guide")
    st.markdown("""
    **Report Types:**
    
    1. **Full Analysis Report**
       - Complete model comparison
       - All metrics and visualizations
       - Cross validation results
       - Recommendations
    
    2. **Single Prediction Report**
       - Individual participant analysis
       - Prediction results with confidence
       - Personalized recommendations
    
    3. **Batch Summary Report**
       - Upload CSV with predictions
       - Summary statistics
       - Distribution analysis
    
    ---
    ### 📋 What's Included
    
    - **Model Performance Tables** (CSV)
    - **Summary Statistics** (CSV)
    - **Key Insights**
    - **Recommendations**
    
    ---
    ### 💡 Tips
    
    - Download reports as CSV files
    - Full report includes all models
    - Single report includes participant data
    - Batch report requires CSV upload
    """)
    
    st.markdown("---")
    st.caption(f"Body Performance Analytics | Report Generator v1.0")
