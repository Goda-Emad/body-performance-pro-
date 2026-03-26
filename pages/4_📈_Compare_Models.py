"""
Page: Compare Models
--------------------
Compare performance of all trained models with interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.model_loader import get_model_info
from utils.visualizations import create_comparison_chart, create_model_comparison_dashboard
import time
import os

# ============================================
# PAGE CONFIGURATION - يجب أن يكون أول أمر
# ============================================
st.set_page_config(
    page_title="Compare Models | Body Performance Analytics",
    page_icon="📈",
    layout="wide"
)

# ============================================
# LOGO IN SIDEBAR
# ============================================
logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'logo.png')
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
st.sidebar.markdown("---")

# ============================================
# LOAD CUSTOM CSS
# ============================================
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
if os.path.exists(css_path):
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.write(f"⚠️ Could not load CSS: {e}")

# العنوان
st.markdown("""
<div class="main-header">
    <h1>📈 Model Performance Comparison</h1>
    <p>Compare all trained models side by side</p>
</div>
""", unsafe_allow_html=True)

# بيانات النماذج (من نتائج التدريب)
CLASSIFICATION_RESULTS = {
    'KNN (k=9)': {
        'Accuracy': 0.6316,
        'Precision': 0.6510,
        'Recall': 0.6316,
        'F1 Score': 0.6352
    },
    'Decision Tree': {
        'Accuracy': 0.6745,
        'Precision': 0.6945,
        'Recall': 0.6745,
        'F1 Score': 0.6746
    },
    'SVM-Linear': {
        'Accuracy': 0.6114,
        'Precision': 0.6113,
        'Recall': 0.6114,
        'F1 Score': 0.6103
    },
    'SVM-RBF': {
        'Accuracy': 0.6887,
        'Precision': 0.6968,
        'Recall': 0.6887,
        'F1 Score': 0.6904
    },
    'Neural Network (MLP)': {
        'Accuracy': 0.7215,
        'Precision': 0.7300,
        'Recall': 0.7215,
        'F1 Score': 0.7231
    }
}

REGRESSION_RESULTS = {
    'Linear Regression': {
        'R²': 0.7658,
        'RMSE': 19.28,
        'MAE': 15.12,
        'MSE': 371.8
    },
    'Decision Tree Regressor': {
        'R²': 0.7221,
        'RMSE': 21.00,
        'MAE': 16.45,
        'MSE': 441.0
    },
    'SVR': {
        'R²': 0.7749,
        'RMSE': 18.90,
        'MAE': 14.89,
        'MSE': 357.2
    },
    'MLP Regressor': {
        'R²': 0.7791,
        'RMSE': 18.73,
        'MAE': 14.72,
        'MSE': 350.8
    }
}

# CROSS VALIDATION RESULTS
CV_RESULTS = {
    'KNN': {'mean': 0.6345, 'std': 0.0165},
    'Decision Tree': {'mean': 0.6723, 'std': 0.0178},
    'SVM-Linear': {'mean': 0.6112, 'std': 0.0152},
    'SVM-RBF': {'mean': 0.6912, 'std': 0.0152},
    'Neural Network (MLP)': {'mean': 0.7298, 'std': 0.0139}
}

# تبويب الصفحة
tab1, tab2, tab3, tab4 = st.tabs(["📊 Classification", "📈 Regression", "🔄 Cross Validation", "🏆 Summary"])

# ============================================
# TAB 1: CLASSIFICATION
# ============================================
with tab1:
    st.markdown("### 📊 Classification Models Performance")
    st.markdown("Comparing 5 classification models on 80:20 train-test split")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        metrics = st.multiselect(
            "Select metrics to display",
            ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            default=['Accuracy', 'F1 Score']
        )
    
    with col2:
        chart_type = st.radio(
            "Chart Type",
            ["Bar Chart", "Radar Chart"],
            horizontal=True
        )
    
    st.markdown("#### 📋 Performance Table")
    clf_df = pd.DataFrame(CLASSIFICATION_RESULTS).T
    st.dataframe(clf_df.style.format("{:.2%}"), use_container_width=True)
    
    st.markdown("#### 📊 Visual Comparison")
    
    if chart_type == "Bar Chart":
        for metric in metrics:
            fig = create_comparison_chart(CLASSIFICATION_RESULTS, metric, f"{metric} Comparison")
            st.plotly_chart(fig, use_container_width=True)
    else:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        fig = go.Figure()
        for model in CLASSIFICATION_RESULTS.keys():
            values = [CLASSIFICATION_RESULTS[model][m] for m in categories]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model,
                line=dict(width=2)
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%')),
            title="Radar Chart - All Classification Models",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    best_model = max(CLASSIFICATION_RESULTS.keys(), key=lambda x: CLASSIFICATION_RESULTS[x]['Accuracy'])
    best_acc = CLASSIFICATION_RESULTS[best_model]['Accuracy']
    
    st.markdown("---")
    st.markdown(f"""
    <div class="main-card" style="text-align: center; background: linear-gradient(135deg, rgba(0,163,196,0.2) 0%, rgba(14,17,23,0) 100%);">
        <div style="font-size: 2rem; margin-bottom: 10px;">🏆</div>
        <h3 style="color: var(--st-color-primary); margin-bottom: 10px;">Best Classification Model</h3>
        <p style="font-size: 1.2rem; font-weight: bold; color: var(--st-color-text);">{best_model}</p>
        <p>Accuracy: <strong>{best_acc:.1%}</strong> | F1 Score: <strong>{CLASSIFICATION_RESULTS[best_model]['F1 Score']:.1%}</strong></p>
        <p style="font-size: 0.9rem; margin-top: 10px;">This model outperforms others by capturing non-linear relationships in the data.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# TAB 2: REGRESSION
# ============================================
with tab2:
    st.markdown("### 📈 Regression Models Performance")
    st.markdown("Comparing 4 regression models predicting broad_jump_cm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Performance Table")
        reg_df = pd.DataFrame(REGRESSION_RESULTS).T
        st.dataframe(reg_df.style.format("{:.4f}" if 'R²' in reg_df.columns else "{:.1f}"), use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 RMSE Comparison (Lower is Better)")
        fig = go.Figure(data=[
            go.Bar(
                x=list(REGRESSION_RESULTS.keys()),
                y=[v['RMSE'] for v in REGRESSION_RESULTS.values()],
                marker_color=['#94a3b8', '#94a3b8', '#94a3b8', '#00A3C4'],
                text=[f"{v['RMSE']:.1f}" for v in REGRESSION_RESULTS.values()],
                textposition='outside'
            )
        ])
        fig.update_layout(
            yaxis_title="RMSE (cm)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 📊 R² Score Comparison (Higher is Better)")
    fig = go.Figure(data=[
        go.Bar(
            x=list(REGRESSION_RESULTS.keys()),
            y=[v['R²'] for v in REGRESSION_RESULTS.values()],
            marker_color=['#94a3b8', '#94a3b8', '#94a3b8', '#00A3C4'],
            text=[f"{v['R²']:.3f}" for v in REGRESSION_RESULTS.values()],
            textposition='outside'
        )
    ])
    fig.update_layout(
        yaxis_title="R² Score",
        yaxis_range=[0, 1],
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    best_reg = max(REGRESSION_RESULTS.keys(), key=lambda x: REGRESSION_RESULTS[x]['R²'])
    best_r2 = REGRESSION_RESULTS[best_reg]['R²']
    
    st.markdown("---")
    st.markdown(f"""
    <div class="main-card" style="text-align: center; background: linear-gradient(135deg, rgba(0,163,196,0.2) 0%, rgba(14,17,23,0) 100%);">
        <div style="font-size: 2rem; margin-bottom: 10px;">🏆</div>
        <h3 style="color: var(--st-color-primary); margin-bottom: 10px;">Best Regression Model</h3>
        <p style="font-size: 1.2rem; font-weight: bold; color: var(--st-color-text);">{best_reg}</p>
        <p>R² Score: <strong>{best_r2:.3f}</strong> | RMSE: <strong>{REGRESSION_RESULTS[best_reg]['RMSE']:.1f} cm</strong></p>
        <p style="font-size: 0.9rem; margin-top: 10px;">This model explains {best_r2:.1%} of variance in broad jump distance.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# TAB 3: CROSS VALIDATION
# ============================================
with tab3:
    st.markdown("### 🔄 Cross Validation Results")
    st.markdown("10-Fold Cross Validation mean accuracy with standard deviation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        models_cv = list(CV_RESULTS.keys())
        means = [CV_RESULTS[m]['mean'] for m in models_cv]
        stds = [CV_RESULTS[m]['std'] for m in models_cv]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=models_cv,
            y=means,
            error_y=dict(type='data', array=stds, visible=True),
            marker_color=['#94a3b8', '#94a3b8', '#94a3b8', '#94a3b8', '#00A3C4'],
            text=[f"{m:.1%}" for m in means],
            textposition='outside'
        ))
        fig.update_layout(
            title="10-Fold Cross Validation Accuracy",
            yaxis_title="Mean Accuracy",
            yaxis_tickformat=".0%",
            yaxis_range=[0.5, 0.8],
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cv_df = pd.DataFrame([
            {'Model': m, 'Mean Accuracy': f"{CV_RESULTS[m]['mean']:.1%}", 
             'Std Dev': f"±{CV_RESULTS[m]['std']:.1%}"}
            for m in models_cv
        ])
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📝 Stability Analysis")
    
    best_cv = max(CV_RESULTS.keys(), key=lambda x: CV_RESULTS[x]['mean'])
    best_mean = CV_RESULTS[best_cv]['mean']
    best_std = CV_RESULTS[best_cv]['std']
    
    st.markdown(f"""
    - **Most Stable Model**: {best_cv} with mean accuracy {best_mean:.1%} ± {best_std:.1%}
    - **Lowest Variance**: MLP shows the lowest standard deviation (1.39%), indicating consistent performance
    - **Interpretation**: Models with low standard deviation are more reliable for production deployment
    """)
    
    st.markdown("#### 📊 Accuracy Distribution")
    fig = go.Figure()
    for model in models_cv:
        np.random.seed(42)
        simulated = np.random.normal(CV_RESULTS[model]['mean'], CV_RESULTS[model]['std'], 100)
        fig.add_trace(go.Box(
            y=simulated,
            name=model,
            boxmean='sd',
            marker_color='#00A3C4' if model == best_cv else '#94a3b8'
        ))
    fig.update_layout(
        title="Cross Validation Accuracy Distribution",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 4: SUMMARY
# ============================================
with tab4:
    st.markdown("### 🏆 Performance Summary")
    
    best_clf = max(CLASSIFICATION_RESULTS.keys(), key=lambda x: CLASSIFICATION_RESULTS[x]['Accuracy'])
    best_reg = max(REGRESSION_RESULTS.keys(), key=lambda x: REGRESSION_RESULTS[x]['R²'])
    best_cv_model = max(CV_RESULTS.keys(), key=lambda x: CV_RESULTS[x]['mean'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="main-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🏆</div>
            <h3 style="color: var(--st-color-primary);">Classification</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: var(--st-color-text);">{best_clf}</p>
            <p>Accuracy: <b>{CLASSIFICATION_RESULTS[best_clf]['Accuracy']:.1%}</b></p>
            <p>F1 Score: <b>{CLASSIFICATION_RESULTS[best_clf]['F1 Score']:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="main-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">📈</div>
            <h3 style="color: var(--st-color-primary);">Regression</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: var(--st-color-text);">{best_reg}</p>
            <p>R² Score: <b>{REGRESSION_RESULTS[best_reg]['R²']:.3f}</b></p>
            <p>RMSE: <b>{REGRESSION_RESULTS[best_reg]['RMSE']:.1f} cm</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="main-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🔄</div>
            <h3 style="color: var(--st-color-primary);">Cross Validation</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: var(--st-color-text);">{best_cv_model}</p>
            <p>Mean Acc: <b>{CV_RESULTS[best_cv_model]['mean']:.1%}</b></p>
            <p>Std Dev: <b>±{CV_RESULTS[best_cv_model]['std']:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Key Insights")
    
    insights = [
        "**Neural Network (MLP)** is the best classifier with 72.15% accuracy",
        "**MLP Regressor** is the best regressor with R² = 0.7791",
        "**10-Fold CV** confirms model stability (±1.39% for MLP)",
        "**Flexibility** (r=+0.59) and **Body Fat** (r=-0.34) are strongest predictors",
        "**KNN** performs best with k=9 (63.16% accuracy)",
        "**SVM-RBF** outperforms SVM-Linear (68.87% vs 61.14%)"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.markdown("---")
    st.markdown("### 💡 Recommendation")
    st.markdown("""
    For production deployment, we recommend:
    
    1. **Classification**: Use **MLP Neural Network** with StandardScaler
    2. **Regression**: Use **MLP Regressor** for broad jump prediction
    3. **Validation**: Always use K-Fold Cross Validation for reliable evaluation
    4. **Key Features**: Focus on flexibility and body fat measurements as they are the strongest predictors
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Complete Dashboard")
    
    fig = create_model_comparison_dashboard(CLASSIFICATION_RESULTS, REGRESSION_RESULTS)
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Classification Models (5)**
    - KNN, Decision Tree, SVM-Linear, SVM-RBF, MLP
    
    **Regression Models (4)**
    - Linear, Decision Tree, SVR, MLP Regressor
    
    **Metrics**
    - Accuracy, Precision, Recall, F1
    - R², RMSE, MAE, MSE
    
    **Validation**
    - 80:20 train-test split
    - 10-Fold Cross Validation
    
    ---
    ### 📈 Results Summary
    
    | Task | Best Model | Score |
    |------|------------|-------|
    | Classification | MLP | 72.15% |
    | Regression | MLP Regressor | R²=0.779 |
    | CV Stability | MLP | ±1.39% |
    """)
    
    st.markdown("---")
    st.caption("Body Performance Analytics | Model Comparison")
