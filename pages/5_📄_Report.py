"""
Page: Generate Report
---------------------
Generate comprehensive PDF reports with analysis and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import io
from utils.report_generator import ReportGenerator, generate_summary
from utils.model_loader import get_model_info
from utils.visualizations import create_comparison_chart, create_model_comparison_dashboard
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
    <div class="main-subtitle">Create comprehensive PDF reports with analysis, predictions, and recommendations</div>
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
    generate_btn = st.button("📄 Generate Report", type="primary", use_container_width=True)

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
    with st.spinner("🔄 Generating report... This may take a few moments"):
        try:
            report_gen = ReportGenerator(title="Body Performance Analytics Report")
            
            if report_type == "Single Prediction Report":
                # Prepare model results
                model_results = {
                    'classification': CLASSIFICATION_RESULTS,
                    'regression': REGRESSION_RESULTS
                }
                
                pdf_buffer = report_gen.generate_report(
                    input_data=input_data,
                    predictions=predictions,
                    model_results=model_results,
                    include_charts=include_charts
                )
                
            elif report_type == "Batch Summary Report" and 'batch_df' in locals():
                # Generate batch summary
                summary = {
                    'total_records': len(batch_df),
                    'columns': list(batch_df.columns),
                    'class_distribution': batch_df['predicted_class'].value_counts().to_dict() if 'predicted_class' in batch_df.columns else {},
                    'avg_jump': batch_df['predicted_broad_jump_cm'].mean() if 'predicted_broad_jump_cm' in batch_df.columns else None
                }
                
                # Create a simple report
                pdf_buffer = io.BytesIO()
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet
                
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                story = []
                styles = getSampleStyleSheet()
                
                story.append(Paragraph("Batch Prediction Report", styles['Title']))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"Total Records: {summary['total_records']}", styles['Normal']))
                story.append(Spacer(1, 12))
                
                if summary['class_distribution']:
                    story.append(Paragraph("Class Distribution:", styles['Heading2']))
                    for cls, count in summary['class_distribution'].items():
                        story.append(Paragraph(f"{cls}: {count}", styles['Normal']))
                
                doc.build(story)
                pdf_buffer.seek(0)
                
            else:  # Full Analysis Report
                # Prepare sample input
                sample_input = {
                    'age': 25,
                    'gender': 'Male',
                    'height_cm': 170,
                    'weight_kg': 70,
                    'body fat_%': 20,
                    'diastolic': 80,
                    'systolic': 120,
                    'gripForce': 40,
                    'sit_and_bend_forward_cm': 15,
                    'sit-ups counts': 40
                }
                
                sample_predictions = {
                    'classification': {'predicted_class': 'B', 'confidence': 0.72},
                    'regression': {'predicted_value': 190.5}
                }
                
                model_results = {
                    'classification': CLASSIFICATION_RESULTS,
                    'regression': REGRESSION_RESULTS
                }
                
                pdf_buffer = report_gen.generate_report(
                    input_data=sample_input,
                    predictions=sample_predictions,
                    model_results=model_results,
                    include_charts=include_charts
                )
            
            # Download button
            st.success("✅ Report generated successfully!")
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"body_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
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
    
    - **Model Performance Tables**
    - **Comparison Charts**
    - **Cross Validation Results**
    - **Key Insights**
    - **Recommendations**
    - **Professional Formatting**
    
    ---
    ### 💡 Tips
    
    - Full report takes 10-15 seconds
    - Single report is fastest
    - Batch report requires CSV upload
    - All reports include professional formatting
    """)
    
    st.markdown("---")
    st.caption(f"Body Performance Analytics | Report Generator v1.0")
