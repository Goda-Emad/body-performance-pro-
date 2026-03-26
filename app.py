"""
Body Performance Analytics - Intelligent Classification System
Main Application Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# إعدادات الصفحة
st.set_page_config(
    page_title="Body Performance Analytics",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# HEADER SECTION
# ============================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2c4e6e 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
    <h1 style="color: white; text-align: center;">🏋️ Body Performance Analytics</h1>
    <p style="color: rgba(255,255,255,0.9); text-align: center;">Intelligent Classification & Regression System for Physical Fitness Assessment</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# KEY METRICS DASHBOARD
# ============================================
st.markdown("### 📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", "13,393")
with col2:
    st.metric("Features", "12")
with col3:
    st.metric("Performance Classes", "4")
with col4:
    st.metric("Samples per Class", "~3,348")

# ============================================
# FEATURE CARDS - NAVIGATION
# ============================================
st.markdown("### 🚀 Explore the Application")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🎯 **Individual Prediction**\n\nEnter participant data and get real-time predictions")
    if st.button("Go to Prediction →", use_container_width=True):
        st.switch_page("pages/1_🎯_Predict.py")

with col2:
    st.info("⚙️ **Model Tuning**\n\nAdjust hyperparameters and see real-time performance changes")
    if st.button("Go to Model Tuning →", use_container_width=True):
        st.switch_page("pages/2_⚙️_Model_Tuning.py")

with col3:
    st.info("📊 **Batch Prediction**\n\nUpload CSV files and get predictions for multiple participants")
    if st.button("Go to Batch Predict →", use_container_width=True):
        st.switch_page("pages/3_📊_Batch_Predict.py")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📈 **Compare Models**\n\nCompare performance of all 8 models")
    if st.button("Go to Compare →", use_container_width=True):
        st.switch_page("pages/4_📈_Compare_Models.py")

with col2:
    st.info("📄 **Generate Report**\n\nCreate comprehensive PDF reports")
    if st.button("Go to Report →", use_container_width=True):
        st.switch_page("pages/5_📄_Report.py")

with col3:
    st.info("ℹ️ **About**\n\nLearn about the project, methodology, and team")
    if st.button("Go to About →", use_container_width=True):
        st.switch_page("pages/6_ℹ️_About.py")

# ============================================
# QUICK STATS SECTION
# ============================================
st.markdown("### 📈 Model Performance Highlights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Best Classification Models")
    clf_data = pd.DataFrame({
        'Model': ['MLP', 'SVM-RBF', 'Decision Tree', 'KNN', 'SVM-Linear'],
        'Accuracy (%)': [72.15, 68.87, 67.45, 63.16, 61.14]
    })
    st.dataframe(clf_data, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Best Regression Models")
    reg_data = pd.DataFrame({
        'Model': ['MLP Regressor', 'SVR', 'Linear Regression', 'Decision Tree'],
        'R² Score': [0.7791, 0.7749, 0.7658, 0.7221],
        'RMSE (cm)': [18.73, 18.90, 19.28, 21.00]
    })
    st.dataframe(reg_data, use_container_width=True, hide_index=True)

# ============================================
# KEY PREDICTORS SECTION
# ============================================
st.markdown("### 🔍 Key Performance Predictors")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Flexibility", "r = +0.59", "Strongest predictor")
with col2:
    st.metric("Body Fat %", "r = -0.34", "Strongest negative")
with col3:
    st.metric("Muscular Endurance", "r = +0.45", "Second strongest")
with col4:
    st.metric("Explosive Power", "r = +0.26", "Moderate")

# ============================================
# TEAM SECTION
# ============================================
st.markdown("### 👥 Team Members")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("**Goda EMAD**\nTeam Leader")
with col2:
    st.markdown("**Ahmed Salama**\nML Engineer")
with col3:
    st.markdown("**Alwafa Ashour**\nData Analyst")
with col4:
    st.markdown("**Ibrahim Elshafey**\nBackend Dev")
with col5:
    st.markdown("**Elia Fahmy**\nFrontend Dev")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #64748b;">
    <p>🏋️ Body Performance Analytics and Intelligent Classification System</p>
    <p>Course: Introduction to AI and Machine Learning | Submission Date: March 26, 2026</p>
    <p style="font-size: 0.7rem;">All predictions are for educational purposes only.</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 🏋️ Body Performance Analytics")
    st.markdown("---")
    
    st.markdown("#### 📊 Quick Stats")
    st.metric("Total Models", "8", delta="4 Classification + 4 Regression")
    st.metric("Best Accuracy", "72.15%", delta="MLP")
    st.metric("Best R²", "0.7791", delta="MLP Regressor")
    
    st.markdown("---")
    st.markdown("#### 🚀 Quick Navigation")
    
    pages = {
        "🎯 Predict": "1_🎯_Predict.py",
        "⚙️ Tuning": "2_⚙️_Model_Tuning.py",
        "📊 Batch": "3_📊_Batch_Predict.py",
        "📈 Compare": "4_📈_Compare_Models.py",
        "📄 Report": "5_📄_Report.py",
        "ℹ️ About": "6_ℹ️_About.py"
    }
    
    for name, page in pages.items():
        if st.button(name, key=f"sidebar_{page}", use_container_width=True):
            st.switch_page(f"pages/{page}")
    
    st.markdown("---")
    st.markdown("#### 📚 Resources")
    st.markdown("- [Kaggle Dataset](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)")
    st.markdown("- [scikit-learn Docs](https://scikit-learn.org)")
    st.markdown("- [Streamlit Docs](https://streamlit.io)")
    
    st.markdown("---")
    st.caption(f"© 2026 Body Performance Analytics Team")
    st.caption(f"Last Updated: March 26, 2026")
