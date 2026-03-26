"""
Page: About
-----------
Information about the project, team, and methodology.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# إعدادات الصفحة
st.set_page_config(
    page_title="About | Body Performance Analytics",
    page_icon="ℹ️",
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
    <div class="main-title">ℹ️ About This Project</div>
    <div class="main-subtitle">Body Performance Analytics and Intelligent Classification System</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# PROJECT OVERVIEW
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📋 Project Overview</div>', unsafe_allow_html=True)

st.markdown("""
### 🎯 Objective
This project implements a complete machine learning pipeline to analyze body performance data and predict:
- **Performance Class** (A=Best, B, C, D=Worst) using classification models
- **Broad Jump Distance** (explosive power) using regression models

### 📊 Dataset
The **Body Performance Dataset** contains measurements from **13,393 individuals** who underwent physical fitness evaluation.

| Aspect | Details |
|--------|---------|
| **Records** | 13,393 participants |
| **Features** | 12 attributes |
| **Target (Classification)** | class (A, B, C, D) |
| **Target (Regression)** | broad_jump_cm |
| **Class Balance** | Perfectly balanced (~3,348 per class) |

### 🔬 Features
- **Demographic:** age, gender
- **Body Composition:** height_cm, weight_kg, body_fat_%
- **Vital Signs:** diastolic, systolic
- **Strength:** gripForce
- **Flexibility:** sit_and_bend_forward_cm
- **Endurance:** sit_ups_counts
- **Explosive Power:** broad_jump_cm
""")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TEAM MEMBERS
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">👥 Team Members</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Goda EMAD</div>
        <div style="font-size: 0.8rem; color: #64748b;">Team Leader</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Ahmed Salama</div>
        <div style="font-size: 0.8rem; color: #64748b;">ML Engineer</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Alwafa Ashour</div>
        <div style="font-size: 0.8rem; color: #64748b;">Data Analyst</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Ibrahim Elshafey</div>
        <div style="font-size: 0.8rem; color: #64748b;">Backend Developer</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold;">Elia Fahmy</div>
        <div style="font-size: 0.8rem; color: #64748b;">Frontend Developer</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Course:** Introduction to AI and Machine Learning")
st.markdown("**Submission Date:** March 26, 2026")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# METHODOLOGY
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🔬 Methodology</div>', unsafe_allow_html=True)

st.markdown("""
### 📊 Machine Learning Pipeline

| Stage | Description |
|-------|-------------|
| **1. Data Preparation** | Handle missing values, duplicates, outliers, data validation |
| **2. EDA** | Visualizations, correlations, distribution analysis |
| **3. Preprocessing** | Encoding (gender, class), StandardScaler |
| **4. Model Training** | 4 classification + 4 regression models |
| **5. Cross Validation** | 80:20, 70:30, 50:50 splits + 10-Fold CV |
| **6. Evaluation** | Accuracy, Precision, Recall, F1, MSE, RMSE, R² |

### 🤖 Models Implemented

| Type | Models |
|------|--------|
| **Classification** | KNN, Decision Tree, SVM (Linear & RBF), Neural Network (MLP) |
| **Regression** | Linear Regression, Decision Tree Regressor, SVR, MLP Regressor |

### 📈 Key Findings

- **Best Classifier:** MLP Neural Network — **72.15% accuracy**
- **Best Regressor:** MLP Regressor — **R² = 0.7791**, RMSE = 18.73 cm
- **Strongest Predictor:** Flexibility (r = +0.59)
- **Strongest Negative Predictor:** Body Fat (r = -0.34)
- **10-Fold CV:** MLP achieved **72.98% ± 1.39%**
""")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TECHNICAL STACK
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🛠️ Technical Stack</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📚 Libraries
    
    | Category | Libraries |
    |----------|----------|
    | **Core** | Python 3.12, pandas, numpy |
    | **ML** | scikit-learn, joblib |
    | **Visualization** | plotly, matplotlib, seaborn |
    | **Web App** | streamlit |
    | **Reports** | reportlab |
    """)

with col2:
    st.markdown("""
    ### 🛠️ Tools
    
    | Tool | Purpose |
    |------|---------|
    | **Google Colab** | Model training & development |
    | **GitHub** | Version control & deployment |
    | **Streamlit** | Web application |
    | **Kaggle** | Dataset source |
    """)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# KEY INSIGHTS
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">💡 Key Insights from EDA</div>', unsafe_allow_html=True)

insights = [
    {
        'insight': 'Flexibility is the Strongest Predictor',
        'detail': 'sit_and_bend_forward_cm has the highest correlation with performance class (r = +0.59)',
        'icon': '🏃'
    },
    {
        'insight': 'Body Fat is the Strongest Negative Predictor',
        'detail': 'body_fat_% shows strong negative correlation (r = -0.34)',
        'icon': '🍔'
    },
    {
        'insight': 'Perfectly Balanced Dataset',
        'detail': 'All four classes have approximately 3,348 records each',
        'icon': '⚖️'
    },
    {
        'insight': 'Blood Pressure is a Weak Predictor',
        'detail': 'Diastolic and systolic show near-zero correlation with class',
        'icon': '❤️'
    },
    {
        'insight': 'Gender Influences Performance Indirectly',
        'detail': 'Males excel in strength, females in flexibility',
        'icon': '👥'
    }
]

for insight in insights:
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 0.5rem; background: #f8fafc; border-radius: 10px;">
        <div style="font-size: 2rem; margin-right: 1rem;">{insight['icon']}</div>
        <div>
            <strong>{insight['insight']}</strong><br>
            <span style="color: #64748b;">{insight['detail']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PERFORMANCE SUMMARY
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🏆 Model Performance Summary</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Classification")
    clf_data = {
        'Model': ['KNN', 'Decision Tree', 'SVM-Linear', 'SVM-RBF', 'MLP'],
        'Accuracy': [0.6316, 0.6745, 0.6114, 0.6887, 0.7215]
    }
    clf_df = pd.DataFrame(clf_data)
    st.dataframe(clf_df.style.format({'Accuracy': '{:.1%}'}), use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Regression")
    reg_data = {
        'Model': ['Linear', 'Decision Tree', 'SVR', 'MLP'],
        'R²': [0.7658, 0.7221, 0.7749, 0.7791],
        'RMSE': [19.28, 21.00, 18.90, 18.73]
    }
    reg_df = pd.DataFrame(reg_data)
    st.dataframe(reg_df.style.format({'R²': '{:.3f}', 'RMSE': '{:.1f}'}), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
### 🏆 Best Models

| Task | Best Model | Performance |
|------|------------|-------------|
| **Classification** | MLP Neural Network | **72.15%** accuracy |
| **Regression** | MLP Regressor | **R² = 0.7791**, RMSE = 18.73 cm |
| **Validation** | MLP (10-Fold CV) | **72.98% ± 1.39%** |
""")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# APP FEATURES
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📱 Application Features</div>', unsafe_allow_html=True)

features = [
    "**Individual Prediction** - Enter participant data and get real-time predictions",
    "**Model Tuning** - Adjust hyperparameters (k, max_depth, kernel) and see results",
    "**Batch Prediction** - Upload CSV files and get predictions for multiple participants",
    "**Model Comparison** - Compare all models side by side with visualizations",
    "**Report Generation** - Download comprehensive PDF reports",
    "**Interactive Visualizations** - Dynamic charts with Plotly",
    "**Professional UI** - Clean, responsive design with custom CSS"
]

for feature in features:
    st.markdown(f"- {feature}")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ACKNOWLEDGMENTS
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🙏 Acknowledgments</div>', unsafe_allow_html=True)

st.markdown("""
- **Kaggle** for providing the Body Performance Dataset
- **scikit-learn** for machine learning algorithms
- **Streamlit** for the web application framework
- **Plotly** for interactive visualizations
- **Course Instructor** for guidance and support

---

### 📚 References
- Body Performance Dataset: [Kaggle Link](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)
- scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org)
- Streamlit Documentation: [streamlit.io](https://streamlit.io)

---

### 📄 License
This project is developed for educational purposes as part of the **Introduction to AI and Machine Learning** course.

---

**Version:** 1.0.0  
**Last Updated:** March 26, 2026
""")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 2rem; background: #f8fafc; border-radius: 20px;">
    <p style="color: #64748b;">© 2026 Body Performance Analytics Team | Introduction to AI and Machine Learning</p>
    <p style="color: #94a3b8; font-size: 0.8rem;">All predictions are for educational purposes only.</p>
</div>
""", unsafe_allow_html=True)
