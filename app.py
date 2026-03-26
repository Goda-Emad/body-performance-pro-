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

# ============================================
# PAGE CONFIGURATION - يجب أن يكون أول أمر
# ============================================
st.set_page_config(
    page_title="Body Performance Analytics",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD CUSTOM CSS
# ============================================
def load_css():
    """Load custom CSS from assets folder"""
    css_path = os.path.join('assets', 'style.css')
    if os.path.exists(css_path):
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        except Exception as e:
            st.write(f"⚠️ Could not load CSS: {e}")
    else:
        # CSS احتياطي في حال عدم وجود الملف
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #2c4e6e 100%);
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 0 0 30px 30px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
            text-align: center;
            letter-spacing: -0.02em;
        }
        .main-subtitle {
            font-size: 1.1rem;
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-top: 0.5rem;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e3a5f;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #1e3a5f;
            text-align: center;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #64748b;
            text-align: center;
            margin-top: 0.3rem;
        }
        .footer {
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            background: white;
            border-radius: 20px;
            color: #64748b;
            font-size: 0.85rem;
            border-top: 1px solid #e2e8f0;
        }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ============================================
# LOGO SECTION
# ============================================
logo_path = os.path.join('assets', 'logo.png')
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path, width=100, use_container_width=False)
        st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# HEADER SECTION
# ============================================
st.markdown("""
<div class="main-header">
    <div class="main-title">🏋️ Body Performance Analytics</div>
    <div class="main-subtitle">Intelligent Classification & Regression System for Physical Fitness Assessment</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# KEY METRICS DASHBOARD
# ============================================
st.markdown("### 📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div style="font-size: 2rem;">📊</div>
        <div class="metric-value">13,393</div>
        <div class="metric-label">Total Records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div style="font-size: 2rem;">🎯</div>
        <div class="metric-value">12</div>
        <div class="metric-label">Features</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div style="font-size: 2rem;">🏆</div>
        <div class="metric-value">4</div>
        <div class="metric-label">Performance Classes</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div style="font-size: 2rem;">⚖️</div>
        <div class="metric-value">3,348</div>
        <div class="metric-label">Samples per Class</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FEATURE CARDS - NAVIGATION
# ============================================
st.markdown("### 🚀 Explore the Application")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div class="feature-icon">🎯</div>
        <div class="card-title">Individual Prediction</div>
        <p style="color: #64748b;">Enter participant data and get real-time predictions for performance class and broad jump distance.</p>
        <span class="badge badge-primary">Try Now</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 Go to Prediction", key="btn_predict", use_container_width=True):
        st.switch_page("pages/1_🎯_Predict.py")

with col2:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div class="feature-icon">⚙️</div>
        <div class="card-title">Model Tuning</div>
        <p style="color: #64748b;">Adjust hyperparameters (k, max_depth, kernel) and see real-time performance changes.</p>
        <span class="badge badge-primary">Experiment</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("⚙️ Go to Model Tuning", key="btn_tuning", use_container_width=True):
        st.switch_page("pages/2_⚙️_Model_Tuning.py")

with col3:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div class="feature-icon">📊</div>
        <div class="card-title">Batch Prediction</div>
        <p style="color: #64748b;">Upload CSV files and get predictions for multiple participants at once.</p>
        <span class="badge badge-primary">Upload</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Go to Batch Predict", key="btn_batch", use_container_width=True):
        st.switch_page("pages/3_📊_Batch_Predict.py")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div class="feature-icon">📈</div>
        <div class="card-title">Compare Models</div>
        <p style="color: #64748b;">Compare performance of all 8 models with interactive charts and tables.</p>
        <span class="badge badge-primary">Analyze</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📈 Go to Compare", key="btn_compare", use_container_width=True):
        st.switch_page("pages/4_📈_Compare_Models.py")

with col2:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div class="feature-icon">📄</div>
        <div class="card-title">Generate Report</div>
        <p style="color: #64748b;">Create comprehensive PDF reports with analysis, predictions, and recommendations.</p>
        <span class="badge badge-primary">Download</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📄 Go to Report", key="btn_report", use_container_width=True):
        st.switch_page("pages/5_📄_Report.py")

with col3:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div class="feature-icon">ℹ️</div>
        <div class="card-title">About</div>
        <p style="color: #64748b;">Learn about the project, methodology, team, and technical stack.</p>
        <span class="badge badge-primary">Info</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("ℹ️ Go to About", key="btn_about", use_container_width=True):
        st.switch_page("pages/6_ℹ️_About.py")

# ============================================
# QUICK STATS SECTION
# ============================================
st.markdown("### 📈 Model Performance Highlights")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏆 Best Classification Models</div>', unsafe_allow_html=True)
    
    clf_data = {
        'Model': ['MLP (Neural Network)', 'SVM-RBF', 'Decision Tree', 'KNN', 'SVM-Linear'],
        'Accuracy': [72.15, 68.87, 67.45, 63.16, 61.14]
    }
    clf_df = pd.DataFrame(clf_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=clf_df['Model'],
            y=clf_df['Accuracy'],
            marker_color=['#1e3a5f', '#2c4e6e', '#4682b4', '#6c8ebf', '#94a3b8'],
            text=[f"{v:.1f}%" for v in clf_df['Accuracy']],
            textposition='outside',
            hovertemplate='%{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
        )
    ])
    fig.update_layout(
        title="Classification Model Accuracy Comparison",
        yaxis_title="Accuracy (%)",
        yaxis_range=[50, 80],
        height=350,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Best Regression Models</div>', unsafe_allow_html=True)
    
    reg_data = {
        'Model': ['MLP Regressor', 'SVR', 'Linear Regression', 'Decision Tree'],
        'R²': [0.7791, 0.7749, 0.7658, 0.7221]
    }
    reg_df = pd.DataFrame(reg_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=reg_df['Model'],
            y=reg_df['R²'],
            marker_color=['#1e3a5f', '#2c4e6e', '#4682b4', '#94a3b8'],
            text=[f"{v:.3f}" for v in reg_df['R²']],
            textposition='outside',
            hovertemplate='%{x}<br>R² Score: %{y:.3f}<extra></extra>'
        )
    ])
    fig.update_layout(
        title="Regression Model R² Score Comparison",
        yaxis_title="R² Score",
        yaxis_range=[0.6, 0.85],
        height=350,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# KEY PREDICTORS SECTION
# ============================================
st.markdown("### 🔍 Key Performance Predictors")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card" style="text-align: center; border-top: 4px solid #10b981;">
        <div style="font-size: 2rem;">🏃</div>
        <div class="metric-value">r = +0.59</div>
        <div class="metric-label">Flexibility</div>
        <p style="font-size: 0.8rem; color: #64748b;">Strongest positive predictor</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card" style="text-align: center; border-top: 4px solid #ef4444;">
        <div style="font-size: 2rem;">🍔</div>
        <div class="metric-value">r = -0.34</div>
        <div class="metric-label">Body Fat %</div>
        <p style="font-size: 0.8rem; color: #64748b;">Strongest negative predictor</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card" style="text-align: center; border-top: 4px solid #f59e0b;">
        <div style="font-size: 2rem;">💪</div>
        <div class="metric-value">r = +0.45</div>
        <div class="metric-label">Muscular Endurance</div>
        <p style="font-size: 0.8rem; color: #64748b;">Second strongest predictor</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card" style="text-align: center; border-top: 4px solid #3b82f6;">
        <div style="font-size: 2rem;">⚡</div>
        <div class="metric-value">r = +0.26</div>
        <div class="metric-label">Explosive Power</div>
        <p style="font-size: 0.8rem; color: #64748b;">Moderate positive correlation</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# TEAM SECTION
# ============================================
st.markdown("### 👥 Team Members")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Goda EMAD</div>
        <div style="font-size: 0.7rem; color: #64748b;">Team Leader</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Ahmed Salama</div>
        <div style="font-size: 0.7rem; color: #64748b;">ML Engineer</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Alwafa Ashour</div>
        <div style="font-size: 0.7rem; color: #64748b;">Data Analyst</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Ibrahim Elshafey</div>
        <div style="font-size: 0.7rem; color: #64748b;">Backend Dev</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Elia Fahmy</div>
        <div style="font-size: 0.7rem; color: #64748b;">Frontend Dev</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <p>🏋️ Body Performance Analytics and Intelligent Classification System</p>
    <p>Course: Introduction to AI and Machine Learning | Submission Date: March 26, 2026</p>
    <p style="font-size: 0.7rem;">All predictions are for educational purposes only. This system should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    # Logo in sidebar
    logo_path = os.path.join('assets', 'logo.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=60)
    
    st.markdown("### 🏋️ Body Performance Analytics")
    st.markdown("---")
    
    st.markdown("#### 📊 Quick Stats")
    st.metric("Total Models", "8", delta="4 Classification + 4 Regression")
    st.metric("Best Accuracy", "72.15%", delta="MLP Neural Network")
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
