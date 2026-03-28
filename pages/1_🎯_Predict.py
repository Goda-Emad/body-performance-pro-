
"""
Page: Individual Prediction
---------------------------
Allows users to input participant data and get predictions from all models.
"""

import streamlit as st

# ============================================
# PAGE CONFIGURATION - يجب أن يكون أول أمر
# ============================================
st.set_page_config(
    page_title="Predict | Body Performance Analytics",
    page_icon="🎯",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# IMPORT UTILITIES
# ============================================
try:
    from utils.model_loader import load_models
    from utils.prediction import predict_classification
    UTILS_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Some utilities not available: {e}")
    UTILS_AVAILABLE = False

# ============================================
# LOAD CUSTOM CSS
# ============================================
def load_css():
    css_path = os.path.join('assets', 'style.css')
    if os.path.exists(css_path):
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        except:
            pass
    else:
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #2c4e6e 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
        }
        .main-title {
            font-size: 2rem;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        .main-subtitle {
            color: rgba(255,255,255,0.9);
            text-align: center;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .card-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1e3a5f;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <div class="main-title">🎯 Individual Performance Prediction</div>
    <div class="main-subtitle">Enter participant data to predict performance class and broad jump distance</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# STATISTICS FROM NOTEBOOK (محدثة بأرقام Random Forest)
# ============================================
DATASET_STATS = {
    'broad_jump_mean': 190.13,
    'broad_jump_std': 39.87,
    'broad_jump_min': 71.0,
    'broad_jump_max': 303.0,
    'accuracy_mlp': 0.7215,        # MLP للتصنيف
    'rmse_rf': 14.0094,            # Random Forest Regressor (أفضل)
    'r2_rf': 0.8757                # Random Forest Regressor (أفضل)
}

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def get_models():
    """Load all trained models with caching"""
    if not UTILS_AVAILABLE:
        return None
    try:
        models = load_models()
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

models = get_models()

# ============================================
# INPUT FORM
# ============================================
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Participant Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=25, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
        body_fat = st.number_input("Body Fat (%)", min_value=3.0, max_value=50.0, value=20.0, step=0.5)
    
    with col2:
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80, step=1)
        systolic = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=200, value=120, step=1)
        grip_force = st.number_input("Grip Force (kg)", min_value=10, max_value=100, value=40, step=1)
        flexibility = st.number_input("Sit and Bend Forward (cm)", min_value=-20, max_value=50, value=15, step=1)
        sit_ups = st.number_input("Sit-ups Count", min_value=0, max_value=100, value=40, step=1)
    
    input_data = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'height_cm': height,
        'weight_kg': weight,
        'body fat_%': body_fat,
        'diastolic': diastolic,
        'systolic': systolic,
        'gripForce': grip_force,
        'sit and bend forward_cm': flexibility,
        'sit-ups counts': sit_ups
    }
    
    with st.expander("📋 Data Summary"):
        summary_df = pd.DataFrame([input_data]).T.reset_index()
        summary_df.columns = ['Feature', 'Value']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Performance", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION FUNCTIONS
# ============================================
def preprocess_input(input_data, scaler):
    """Preprocess single input sample keeping EXACT Colab feature names and order"""
    
    colab_feature_order = [
        'age', 
        'height_cm', 
        'weight kg',                 # مسافة مش شرطة
        'body fat_%', 
        'diastolic', 
        'systolic', 
        'gripForce', 
        'sit and bend forward cm',   # مسافة مش شرطة
        'sit-ups counts',
        'gender_num'
    ]
    
    mapped_data = {
        'age': input_data['age'],
        'height_cm': input_data['height_cm'],
        'weight kg': input_data['weight_kg'],
        'body fat_%': input_data['body fat_%'],
        'diastolic': input_data['diastolic'],
        'systolic': input_data['systolic'],
        'gripForce': input_data['gripForce'],
        'sit and bend forward cm': input_data['sit and bend forward_cm'],
        'sit-ups counts': input_data['sit-ups counts'],
        'gender_num': input_data['gender']
    }
    
    X_df = pd.DataFrame([mapped_data])[colab_feature_order]
    X_scaled = scaler.transform(X_df)
    
    return X_scaled

def get_class_description(class_label):
    """Get description for performance class"""
    descriptions = {
        'A': {
            'name': 'Excellent Performance',
            'description': 'Top 25% of participants. Exceptional fitness level.',
            'color': '#1e3a5f',
            'icon': '🏆',
            'recommendation': 'Maintain current training regimen.'
        },
        'B': {
            'name': 'Good Performance',
            'description': 'Above average fitness level. Strong performance.',
            'color': '#10b981',
            'icon': '💪',
            'recommendation': 'Continue consistent training. Small improvements possible.'
        },
        'C': {
            'name': 'Average Performance',
            'description': 'Average fitness level. Room for improvement.',
            'color': '#f59e0b',
            'icon': '📊',
            'recommendation': 'Focus on flexibility and endurance training.'
        },
        'D': {
            'name': 'Needs Improvement',
            'description': 'Below average fitness level. Significant improvement needed.',
            'color': '#ef4444',
            'icon': '⚠️',
            'recommendation': 'Consider structured fitness program focusing on core areas.'
        }
    }
    return descriptions.get(class_label, descriptions['D'])

def get_regression_interpretation(pred_value):
    """Get interpretation based on prediction compared to dataset distribution"""
    mean = DATASET_STATS['broad_jump_mean']
    std = DATASET_STATS['broad_jump_std']
    
    if pred_value > mean + std:
        return "🚀 Excellent explosive power! Elite level performance (top 16%)."
    elif pred_value > mean:
        return "💪 Above average explosive power. Good athletic capability."
    elif pred_value > mean - std:
        return "📊 Average explosive power. Can be improved with plyometric training."
    else:
        return "📉 Below average explosive power. Consider targeted strength training."

def predict_regression_safe(model, X_scaled):
    """Safe prediction function for regression"""
    try:
        return model.predict(X_scaled)
    except Exception as e:
        st.warning(f"Regression prediction error: {e}")
        return np.array([DATASET_STATS['broad_jump_mean']])

# ============================================
# MAKE PREDICTION
# ============================================
if predict_btn:
    with st.spinner("🔄 Making predictions..."):
        try:
            if models is None or not UTILS_AVAILABLE:
                st.warning("⚠️ Models not loaded. Showing demo predictions based on dataset statistics.")
                
                with col_results:
                    # Demo Classification
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">🏆 Classification Results (Demo)</div>', unsafe_allow_html=True)
                    
                    if flexibility > 20:
                        pred_class = 'B'
                        confidence = 0.68
                    elif flexibility > 10:
                        pred_class = 'C'
                        confidence = 0.55
                    else:
                        pred_class = 'D'
                        confidence = 0.62
                    
                    class_desc = get_class_description(pred_class)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, {class_desc['color']}20 0%, white 100%); border-radius: 15px;">
                            <div style="font-size: 3rem;">{class_desc['icon']}</div>
                            <div style="font-size: 2rem; font-weight: bold; color: {class_desc['color']};">{pred_class}</div>
                            <div style="font-size: 0.9rem; color: #64748b;">Predicted Class</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
                            <div style="font-size: 1.2rem; font-weight: bold;">{class_desc['name']}</div>
                            <div style="font-size: 0.8rem; color: #64748b;">Performance Level</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #1e3a5f;">{confidence:.1%}</div>
                            <div style="font-size: 0.8rem; color: #64748b;">Model Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Interpretation:** {class_desc['description']}")
                    st.markdown(f"**Recommendation:** {class_desc['recommendation']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Demo Regression
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">📈 Regression Results (Demo)</div>', unsafe_allow_html=True)
                    
                    pred_jump = 150 + (grip_force * 0.5) + (flexibility * 1.5)
                    pred_jump = np.clip(pred_jump, 100, 280)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
                            <div style="font-size: 2rem; font-weight: bold; color: #1e3a5f;">{pred_jump:.1f}</div>
                            <div style="font-size: 0.8rem; color: #64748b;">Predicted Broad Jump (cm)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    avg_jump = DATASET_STATS['broad_jump_mean']
                    diff = pred_jump - avg_jump
                    with col2:
                        st.metric("Dataset Average", f"{avg_jump:.1f} cm", delta=f"{diff:+.1f} cm")
                    
                    interpretation = get_regression_interpretation(pred_jump)
                    st.markdown(f"**Interpretation:** {interpretation}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                # REAL PREDICTION WITH MODELS
                scaler = models.get('scaler')
                if scaler is None:
                    st.error("❌ Scaler not found in models")
                    st.stop()
                
                X_scaled = preprocess_input(input_data, scaler)
                
                with col_results:
                    # ========== CLASSIFICATION RESULTS ==========
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">🏆 Classification Results</div>', unsafe_allow_html=True)
                    
                    mlp_model = models.get('mlp')
                    pred_out = None
                    proba = None
                    
                    if mlp_model is not None:
                        try:
                            pred_out, proba = predict_classification(mlp_model, X_scaled, return_proba=True)
                        except TypeError:
                            pred_out = predict_classification(mlp_model, X_scaled)
                            if hasattr(mlp_model, "predict_proba"):
                                proba = mlp_model.predict_proba(X_scaled)
                        except Exception as e:
                            st.warning(f"Prediction warning: {e}")
                            if hasattr(mlp_model, "predict"):
                                pred_out = mlp_model.predict(X_scaled)
                                if hasattr(mlp_model, "predict_proba"):
                                    proba = mlp_model.predict_proba(X_scaled)
                    
                    if pred_out is None:
                        pred_out = [0]
                        proba = None
                    
                    confidence = np.max(proba) if proba is not None else None
                    
                    class_map = {3: 'A', 2: 'B', 1: 'C', 0: 'D'}
                    raw_pred = pred_out[0]
                    if isinstance(raw_pred, (int, np.integer, float, np.floating)):
                        final_class = class_map.get(int(raw_pred), 'D')
                    else:
                        final_class = str(raw_pred).upper()
                    
                    class_desc = get_class_description(final_class)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, {class_desc['color']}20 0%, white 100%); border-radius: 15px;">
                            <div style="font-size: 3rem;">{class_desc['icon']}</div>
                            <div style="font-size: 2rem; font-weight: bold; color: {class_desc['color']};">{final_class}</div>
                            <div style="font-size: 0.9rem; color: #64748b;">Predicted Class</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
                            <div style="font-size: 1.2rem; font-weight: bold;">{class_desc['name']}</div>
                            <div style="font-size: 0.8rem; color: #64748b;">Performance Level</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        if confidence:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
                                <div style="font-size: 1.5rem; font-weight: bold; color: #1e3a5f;">{confidence:.1%}</div>
                                <div style="font-size: 0.8rem; color: #64748b;">Model Confidence</div>
                                <div style="font-size: 0.7rem; color: #94a3b8;">(MLP Accuracy: {DATASET_STATS['accuracy_mlp']:.1%})</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if proba is not None and len(proba) > 0:
                        st.markdown("---")
                        st.markdown("**Probability Distribution Across Classes**")
                        prob_df = pd.DataFrame({
                            'Class': ['D', 'C', 'B', 'A'],
                            'Probability': proba[0] * 100
                        })
                        fig = go.Figure(data=[
                            go.Bar(
                                x=prob_df['Class'],
                                y=prob_df['Probability'],
                                marker_color=['#ef4444', '#f59e0b', '#10b981', '#1e3a5f'],
                                text=prob_df['Probability'].round(1),
                                textposition='outside'
                            )
                        ])
                        fig.update_layout(
                            yaxis_title="Probability (%)",
                            yaxis_range=[0, 100],
                            height=300,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"**Interpretation:** {class_desc['description']}")
                    st.markdown(f"**Recommendation:** {class_desc['recommendation']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ========== REGRESSION RESULTS (باستخدام Random Forest) ==========
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">📈 Regression Results (Random Forest)</div>', unsafe_allow_html=True)
                    
                    # ✅ التعديل المهم: استخدام Random Forest Regressor بدل MLP
                    rf_reg = models.get('rf_regressor')  # أو الاسم اللي عندك في load_models
                    
                    if rf_reg is not None:
                        pred_value = predict_regression_safe(rf_reg, X_scaled)
                        predicted_jump = pred_value[0] if len(pred_value) > 0 else DATASET_STATS['broad_jump_mean']
                        model_used = "Random Forest"
                    else:
                        # Fallback: حاول تجيب mlp_regressor لو rf_reg مش موجود
                        mlp_reg = models.get('mlp_regressor')
                        if mlp_reg is not None:
                            pred_value = predict_regression_safe(mlp_reg, X_scaled)
                            predicted_jump = pred_value[0] if len(pred_value) > 0 else DATASET_STATS['broad_jump_mean']
                            model_used = "MLP (Fallback)"
                            st.warning("⚠️ Random Forest model not found, using MLP Regressor as fallback.")
                        else:
                            predicted_jump = DATASET_STATS['broad_jump_mean']
                            model_used = "Dataset Average"
                            st.warning("⚠️ No regression model found, showing dataset average.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
                            <div style="font-size: 2rem; font-weight: bold; color: #1e3a5f;">{predicted_jump:.1f}</div>
                            <div style="font-size: 0.8rem; color: #64748b;">Predicted Broad Jump (cm)</div>
                            <div style="font-size: 0.7rem; color: #94a3b8;">Model: {model_used}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    avg_jump = DATASET_STATS['broad_jump_mean']
                    diff = predicted_jump - avg_jump
                    with col2:
                        st.metric("Dataset Average", f"{avg_jump:.1f} cm", delta=f"{diff:+.1f} cm")
                    
                    interpretation = get_regression_interpretation(predicted_jump)
                    st.markdown(f"**Interpretation:** {interpretation}")
                    
                    # عرض أداء النموذج من الـ Notebook
                    st.markdown(f"**Model Performance (Random Forest):** RMSE = **{DATASET_STATS['rmse_rf']:.2f} cm**, R² = **{DATASET_STATS['r2_rf']:.4f}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error making predictions: {e}")
            st.info("Please check your input data and try again.")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool uses machine learning models trained on **13,393 participants** to predict:
    
    **Classification**
    - **A (Best)** - Top 25% performance
    - **B** - Above average
    - **C** - Average
    - **D (Worst)** - Below average
    
    **Regression (Random Forest)**
    - Predicts **broad jump distance** in cm
    - **R² = 0.8757** (Explains 87.6% of variance)
    - **RMSE = 14.0 cm** (High accuracy)
    
    ---
    ### 🏆 Best Models
    - **Classification:** MLP Neural Network (72.15% accuracy)
    - **Regression:** Random Forest Regressor (R² = 0.8757)
    
    ---
    ### 📊 Key Predictors
    - Flexibility (r = +0.59)
    - Sit-ups (r = +0.45)
    - Body Fat (r = -0.34)
    - Grip Force (r = +0.48)
    """)
    
    st.markdown("---")
    st.caption("Body Performance Analytics | March 2026")
