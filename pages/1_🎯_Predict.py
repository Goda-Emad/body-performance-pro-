"""
Page: Individual Prediction
---------------------------
Allows users to input participant data and get predictions from all models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_models
from utils.prediction import predict_classification, predict_regression
from utils.report_generator import generate_summary
import io

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Predict | Body Performance Analytics",
    page_icon="🎯",
    layout="wide"
)

# ============================================
# LOAD CUSTOM CSS
# ============================================
def load_css():
    """Load custom CSS from assets folder"""
    css_path = os.path.join('assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    else:
        # Fallback CSS if file not found
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
# LOAD MODELS
# ============================================
@st.cache_resource
def get_models():
    """Load all trained models with caching"""
    try:
        models = load_models()
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Please ensure model files are in the 'models/' folder")
        return None

models = get_models()
if models is None:
    st.stop()

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
    
    # Collect input data
    input_data = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'height_cm': height,
        'weight_kg': weight,
        'body fat_%': body_fat,
        'diastolic': diastolic,
        'systolic': systolic,
        'gripForce': grip_force,
        'sit_and_bend_forward_cm': flexibility,
        'sit-ups counts': sit_ups
    }
    
    # Data summary expander
    with st.expander("📋 Data Summary"):
        summary_df = pd.DataFrame([input_data]).T.reset_index()
        summary_df.columns = ['Feature', 'Value']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Performance", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION FUNCTION
# ============================================
def preprocess_input(input_data, scaler):
    """Preprocess single input sample"""
    feature_cols = [
        'age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
        'diastolic', 'systolic', 'gripForce', 'sit_and_bend_forward_cm', 'sit-ups counts'
    ]
    X = np.array([[input_data[col] for col in feature_cols]])
    X_scaled = scaler.transform(X)
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

# ============================================
# MAKE PREDICTION
# ============================================
if predict_btn:
    with st.spinner("🔄 Making predictions..."):
        try:
            # Get scaler
            scaler = models['scaler']
            
            # Preprocess input
            X_scaled = preprocess_input(input_data, scaler)
            
            with col_results:
                # ========== CLASSIFICATION RESULTS ==========
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🏆 Classification Results</div>', unsafe_allow_html=True)
                
                # Best classifier (MLP)
                mlp_model = models['mlp']
                pred_class, proba = predict_classification(mlp_model, X_scaled, return_proba=True)
                confidence = np.max(proba) if proba is not None else None
                
                # Get class description
                class_desc = get_class_description(pred_class[0])
                
                # Display class result
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, {class_desc['color']}20 0%, white 100%); border-radius: 15px;">
                        <div style="font-size: 3rem;">{class_desc['icon']}</div>
                        <div style="font-size: 2rem; font-weight: bold; color: {class_desc['color']};">{pred_class[0]}</div>
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
                            <div style="font-size: 0.8rem; color: #64748b;">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Probability distribution chart
                if proba is not None:
                    st.markdown("---")
                    st.markdown("**Probability Distribution**")
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
                
                # ========== REGRESSION RESULTS ==========
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📈 Regression Results</div>', unsafe_allow_html=True)
                
                # Best regressor (MLP Regressor)
                mlp_reg = models['mlp_regressor']
                pred_value = predict_regression(mlp_reg, X_scaled)
                
                # Display regression result
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 15px;">
                        <div style="font-size: 2rem; font-weight: bold; color: #1e3a5f;">{pred_value[0]:.1f}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Predicted Broad Jump (cm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Compare with average
                avg_jump = 190.13
                diff = pred_value[0] - avg_jump
                with col2:
                    st.metric("Average (Dataset)", f"{avg_jump:.1f} cm", delta=f"{diff:+.1f} cm")
                
                # Interpretation based on prediction
                if pred_value[0] > 220:
                    interpretation = "🚀 Excellent explosive power! Elite level performance."
                elif pred_value[0] > 190:
                    interpretation = "💪 Above average explosive power. Good athletic capability."
                elif pred_value[0] > 160:
                    interpretation = "📊 Average explosive power. Can be improved with plyometric training."
                else:
                    interpretation = "📉 Below average explosive power. Consider targeted strength training."
                
                st.markdown(f"**Interpretation:** {interpretation}")
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
    
    **Regression**
    - Predicts **broad jump distance** in cm
    - Range: 71 - 303 cm
    
    ---
    ### 🏆 Best Models
    - **Classification:** MLP Neural Network (72.15% accuracy)
    - **Regression:** MLP Regressor (R² = 0.7791)
    
    ---
    ### 📊 Key Predictors
    - Flexibility (r = +0.59)
    - Muscular Endurance (r = +0.45)
    - Body Fat (r = -0.34)
    """)
    
    st.markdown("---")
    st.caption("Body Performance Analytics | March 2026")
