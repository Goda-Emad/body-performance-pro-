"""
Page: Batch Prediction
----------------------
Allows users to upload CSV files and get predictions for multiple participants.
"""

import streamlit as st

# ============================================
# PAGE CONFIGURATION - يجب أن يكون أول أمر
# ============================================
st.set_page_config(
    page_title="Batch Predict | Body Performance Analytics",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from utils.model_loader import load_models
from utils.preprocessing import preprocess_batch_data, get_feature_names
from utils.prediction import predict_batch, predict_classification
from utils.visualizations import create_distribution_plot, create_comparison_chart
import time
import os

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
    <div class="main-title">📊 Batch Prediction</div>
    <div class="main-subtitle">Upload a CSV file to predict performance for multiple participants</div>
</div>
""", unsafe_allow_html=True)

# تحميل النماذج
@st.cache_resource
def get_models():
    try:
        models = load_models()
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

models = get_models()
if models is None:
    st.stop()

# تقسيم الصفحة
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📁 Upload Data</div>', unsafe_allow_html=True)
    
    # نموذج البيانات المطلوبة
    st.markdown("### Required Columns")
    required_cols = get_feature_names()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Features (10 columns):**")
        for col in get_feature_names():
            st.markdown(f"- `{col}`")
    with col2:
        st.markdown("**Optional:**")
        st.markdown("- `broad jump_cm` (for comparison)")
        st.markdown("- `class` (for validation)")
    
    # تحميل الملف
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv', 'xlsx'],
        help="Upload a file with the required columns. You can download a sample template below."
    )
    
    # تحميل نموذج
    sample_df = pd.DataFrame({
        'age': [25, 30, 35],
        'gender': ['M', 'F', 'M'],
        'height_cm': [175, 165, 180],
        'weight_kg': [70, 60, 85],
        'body fat_%': [20, 25, 22],
        'diastolic': [80, 75, 85],
        'systolic': [120, 115, 130],
        'gripForce': [45, 35, 50],
        'sit and bend forward_cm': [15, 20, 10],
        'sit-ups counts': [45, 40, 35],
        'broad jump_cm': [200, 180, 190],
        'class': ['B', 'C', 'B']
    })
    
    csv = sample_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="sample_data.csv" style="display: inline-block; padding: 0.5rem 1rem; background-color: #1e3a5f; color: white; border-radius: 8px; text-decoration: none; margin-top: 0.5rem;">📥 Download Sample CSV Template</a>', unsafe_allow_html=True)
    
    # إعدادات التنبؤ
    st.markdown("---")
    st.markdown("### ⚙️ Prediction Settings")
    
    prediction_type = st.radio(
        "Prediction Type",
        ["Classification Only", "Regression Only", "Both"],
        horizontal=True
    )
    
    # اختيار النموذج
    if prediction_type in ["Classification Only", "Both"]:
        clf_model = st.selectbox(
            "Classification Model",
            ["MLP (Neural Network) - Best", "SVM-RBF", "SVM-Linear", "Decision Tree", "KNN"],
            index=0
        )
        clf_map = {
            "MLP (Neural Network) - Best": "mlp",
            "SVM-RBF": "svm_rbf",
            "SVM-Linear": "svm_linear",
            "Decision Tree": "dt",
            "KNN": "knn"
        }
    
    if prediction_type in ["Regression Only", "Both"]:
        reg_model = st.selectbox(
            "Regression Model",
            ["MLP Regressor - Best", "SVR", "Linear Regression", "Decision Tree Regressor"],
            index=0
        )
        reg_map = {
            "MLP Regressor - Best": "mlp_regressor",
            "SVR": "svr",
            "Linear Regression": "linear_regression",
            "Decision Tree Regressor": "dt_regressor"
        }
    
    # زر التنفيذ
    predict_btn = st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# معالجة التنبؤ
if uploaded_file and predict_btn:
    with st.spinner("🔄 Processing batch prediction..."):
        try:
            # قراءة الملف
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # التحقق من الأعمدة المطلوبة
            required_cols = get_feature_names()
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.info("Please ensure your file has all required columns. Download the sample template.")
                st.stop()
            
            # Preprocess - تعديل: تمرير target_column بشكل صحيح
            try:
                X_scaled, y_true, warnings = preprocess_batch_data(df, models['scaler'], target_col='broad jump_cm')
            except TypeError:
                # لو الـ function مش بتقبل target_col، جرب بدونها
                X_scaled, warnings = preprocess_batch_data(df, models['scaler'])
                y_true = df['broad jump_cm'] if 'broad jump_cm' in df.columns else None
            
            if warnings:
                for w in warnings:
                    st.warning(f"⚠️ {w}")
            
            # Initialize results dataframe
            results_df = df.copy()
            
            # Classification predictions
            if prediction_type in ["Classification Only", "Both"]:
                clf_key = clf_map[clf_model]
                clf = models[clf_key]
                
                # Get predictions - تعديل: handle return_proba gracefully
                try:
                    pred_classes, probas = predict_classification(clf, X_scaled, return_proba=True)
                except TypeError:
                    # لو الدالة مش بتدعم return_proba، جرب بدونها
                    pred_classes = predict_classification(clf, X_scaled)
                    probas = None
                
                results_df['predicted_class'] = pred_classes
                
                # Add probabilities if available
                if probas is not None and len(probas) > 0:
                    class_names = ['D', 'C', 'B', 'A']
                    for i, name in enumerate(class_names):
                        if i < probas.shape[1]:
                            results_df[f'prob_{name}'] = probas[:, i]
                
                st.success(f"✅ Classification predictions added for {len(df)} samples")
            
            # Regression predictions
            if prediction_type in ["Regression Only", "Both"]:
                reg_key = reg_map[reg_model]
                reg = models[reg_key]
                
                pred_values = reg.predict(X_scaled)
                results_df['predicted_broad_jump_cm'] = pred_values.round(1)
                
                st.success(f"✅ Regression predictions added for {len(df)} samples")
            
            # Display results in right column
            with col_right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📊 Prediction Results</div>', unsafe_allow_html=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(df))
                if prediction_type in ["Classification Only", "Both"]:
                    class_counts = results_df['predicted_class'].value_counts()
                    with col2:
                        st.metric("Classes Predicted", len(class_counts))
                if prediction_type in ["Regression Only", "Both"]:
                    with col3:
                        st.metric("Avg Jump (Pred)", f"{results_df['predicted_broad_jump_cm'].mean():.1f} cm")
                
                st.markdown("---")
                
                # Show results table
                st.markdown("### Results Preview")
                
                # ============================================
                # التعديل المطلوب - إصلاح عرض الأعمدة
                # ============================================
                # عرض أول 6 أعمدة من الميزات (بدون تكرار gender)
                display_cols = get_feature_names()[:6]
                
                if prediction_type in ["Classification Only", "Both"]:
                    display_cols.append('predicted_class')
                if prediction_type in ["Regression Only", "Both"]:
                    display_cols.append('predicted_broad_jump_cm')
                
                st.dataframe(results_df[display_cols].head(20), use_container_width=True)
                
                if len(df) > 20:
                    st.caption(f"Showing first 20 of {len(df)} rows")
                
                # Visualizations
                st.markdown("---")
                st.markdown("### 📈 Visualizations")
                
                # Class distribution
                if prediction_type in ["Classification Only", "Both"] and 'predicted_class' in results_df.columns:
                    fig = px.pie(
                        results_df, 
                        names='predicted_class', 
                        title='Predicted Class Distribution',
                        color_discrete_sequence=['#1e3a5f', '#2c4e6e', '#4682b4', '#94a3b8']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Regression distribution
                if prediction_type in ["Regression Only", "Both"] and 'predicted_broad_jump_cm' in results_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=results_df['predicted_broad_jump_cm'],
                        nbinsx=30,
                        marker_color='#1e3a5f',
                        opacity=0.7
                    ))
                    fig.update_layout(
                        title="Predicted Broad Jump Distribution",
                        xaxis_title="Broad Jump (cm)",
                        yaxis_title="Count",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Comparison with actual if available
                if 'broad jump_cm' in df.columns and prediction_type in ["Regression Only", "Both"] and 'predicted_broad_jump_cm' in results_df.columns:
                    st.markdown("---")
                    st.markdown("### 📊 Prediction vs Actual")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['broad jump_cm'],
                        y=results_df['predicted_broad_jump_cm'],
                        mode='markers',
                        marker=dict(color='#1e3a5f', size=8, opacity=0.6),
                        name='Predictions'
                    ))
                    
                    # Add perfect prediction line
                    max_val = max(results_df['broad jump_cm'].max(), results_df['predicted_broad_jump_cm'].max())
                    fig.add_trace(go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction'
                    ))
                    
                    fig.update_layout(
                        title="Actual vs Predicted Broad Jump",
                        xaxis_title="Actual (cm)",
                        yaxis_title="Predicted (cm)",
                        height=450,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate error metrics
                    errors = results_df['broad jump_cm'] - results_df['predicted_broad_jump_cm']
                    mae = errors.abs().mean()
                    rmse = np.sqrt((errors**2).mean())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Absolute Error", f"{mae:.1f} cm")
                    with col2:
                        st.metric("RMSE", f"{rmse:.1f} cm")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download results
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📥 Download Results</div>', unsafe_allow_html=True)
                
                # CSV download
                csv_output = results_df.to_csv(index=False)
                b64_csv = base64.b64encode(csv_output.encode()).decode()
                st.markdown(f'<a href="data:file/csv;base64,{b64_csv}" download="predictions.csv" style="display: inline-block; padding: 0.5rem 1rem; background-color: #1e3a5f; color: white; border-radius: 8px; text-decoration: none; margin: 0.25rem;">📥 Download CSV Results</a>', unsafe_allow_html=True)
                
                # Excel download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Predictions', index=False)
                excel_data = output.getvalue()
                b64_excel = base64.b64encode(excel_data).decode()
                st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="predictions.xlsx" style="display: inline-block; padding: 0.5rem 1rem; background-color: #2c4e6e; color: white; border-radius: 8px; text-decoration: none; margin: 0.25rem;">📥 Download Excel Results</a>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please check your file format and ensure it matches the sample template.")

elif uploaded_file and not predict_btn:
    with col_right:
        st.info("👈 Click 'Run Batch Prediction' to start")

# Sidebar with info
with st.sidebar:
    st.markdown("### ℹ️ Batch Prediction Guide")
    st.markdown("""
    **How to use:**
    1. Download the sample template
    2. Add your data (keep column names)
    3. Upload the file
    4. Select prediction type and model
    5. Click 'Run Batch Prediction'
    
    ---
    ### 📋 Required Columns
    
    **Features (10):**
    - `age`, `gender`, `height_cm`, `weight_kg`, `body fat_%`
    - `diastolic`, `systolic`, `gripForce`
    - `sit and bend forward_cm`, `sit-ups counts`
    
    **Optional:**
    - `broad jump_cm` (for comparison)
    - `class` (for validation)
    
    ---
    ### 📊 Output
    
    You'll get:
    - Preview table with predictions
    - Distribution charts
    - CSV/Excel download
    """)
    
    st.markdown("---")
    st.caption("Body Performance Analytics | Batch Prediction v1.0")
