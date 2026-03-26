"""
Page: Model Tuning
------------------
Allows users to adjust hyperparameters (k, max_depth, kernel) and see real-time results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_data, preprocess_raw_data
import time

# إعدادات الصفحة
st.set_page_config(
    page_title="Model Tuning | Body Performance Analytics",
    page_icon="⚙️",
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
    <div class="main-title">⚙️ Model Tuning Studio</div>
    <div class="main-subtitle">Adjust hyperparameters and see real-time performance changes</div>
</div>
""", unsafe_allow_html=True)

# تحميل البيانات - ✅ FIXED: use correct column names with spaces
@st.cache_data
def load_and_preprocess():
    df = load_data()
    df_clean = preprocess_raw_data(df)
    
    # ✅ CORRECTED: feature columns with spaces (matches dataset)
    feature_cols = [
        'age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
        'diastolic', 'systolic', 'gripForce', 'sit and bend forward_cm', 'sit-ups counts'
    ]
    
    X = df_clean[feature_cols].values
    y = df_clean['class_encoded'].values
    
    # Split data (80:20 for consistent testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols

try:
    X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess()
    st.success(f"✅ Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# تبويب النماذج
tab1, tab2, tab3, tab4 = st.tabs(["🔍 K-Nearest Neighbors", "🌳 Decision Tree", "⚡ SVM", "🧠 Neural Network"])

# ============================================
# TAB 1: KNN
# ============================================
with tab1:
    st.markdown("### 🔍 K-Nearest Neighbors (KNN)")
    st.markdown("Adjust the number of neighbors (k) and see how accuracy changes.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        k_value = st.slider("Number of Neighbors (k)", min_value=1, max_value=25, value=9, step=2)
        weights = st.selectbox("Weight Function", ["uniform", "distance"])
        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])
        run_btn = st.button("🚀 Train KNN Model", type="primary", use_container_width=True)
    
    with col2:
        if run_btn:
            with st.spinner("Training KNN..."):
                start_time = time.time()
                
                knn = KNeighborsClassifier(n_neighbors=k_value, weights=weights, metric=metric)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                train_time = time.time() - start_time
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a: st.metric("Accuracy", f"{acc:.2%}")
                with col_b: st.metric("Precision", f"{prec:.2%}")
                with col_c: st.metric("Recall", f"{rec:.2%}")
                with col_d: st.metric("F1 Score", f"{f1:.2%}")
                st.info(f"⏱️ Training time: {train_time:.3f} seconds")
                
                # Plot k vs accuracy
                st.markdown("---")
                st.markdown("### 📈 Accuracy vs k Value")
                k_range = range(1, 26, 2)
                accuracies = []
                for k in k_range:
                    knn_temp = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
                    knn_temp.fit(X_train, y_train)
                    accuracies.append(accuracy_score(y_test, knn_temp.predict(X_test)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(k_range), y=accuracies, mode='lines+markers',
                    name='Accuracy', line=dict(color='#1e3a5f', width=2),
                    marker=dict(size=8, color='#2c4e6e')
                ))
                fig.add_vline(x=k_value, line_dash="dash", line_color="red", annotation_text=f"Current k={k_value}")
                fig.update_layout(
                    title="KNN Accuracy vs Number of Neighbors",
                    xaxis_title="k (Number of Neighbors)",
                    yaxis_title="Accuracy",
                    yaxis_tickformat=".0%",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: Decision Tree
# ============================================
with tab2:
    st.markdown("### 🌳 Decision Tree Classifier")
    st.markdown("Adjust tree depth and splitting criteria.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        max_depth = st.slider("Maximum Depth", min_value=1, max_value=20, value=10)
        min_samples_split = st.slider("Minimum Samples to Split", min_value=2, max_value=20, value=2)
        criterion = st.selectbox("Split Criterion", ["gini", "entropy"])
        run_btn_dt = st.button("🚀 Train Decision Tree", type="primary", use_container_width=True)
    
    with col2:
        if run_btn_dt:
            with st.spinner("Training Decision Tree..."):
                start_time = time.time()
                dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                train_time = time.time() - start_time
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a: st.metric("Accuracy", f"{acc:.2%}")
                with col_b: st.metric("Precision", f"{prec:.2%}")
                with col_c: st.metric("Recall", f"{rec:.2%}")
                with col_d: st.metric("F1 Score", f"{f1:.2%}")
                st.info(f"⏱️ Training time: {train_time:.3f} seconds")
                
                # Feature importance
                st.markdown("---")
                st.markdown("### 📊 Feature Importance")
                importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': dt.feature_importances_}).sort_values('Importance', ascending=True)
                fig = go.Figure(data=[go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h', marker_color='#1e3a5f')])
                fig.update_layout(title="Feature Importance", height=400, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: SVM
# ============================================
with tab3:
    st.markdown("### ⚡ Support Vector Machine (SVM)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        kernel = st.selectbox("Kernel Type", ["rbf", "linear", "poly", "sigmoid"])
        C_value = st.slider("Regularization Parameter (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        gamma = st.selectbox("Gamma", ["scale", "auto", 0.1, 1.0])
        run_btn_svm = st.button("🚀 Train SVM", type="primary", use_container_width=True)
    
    with col2:
        if run_btn_svm:
            with st.spinner("Training SVM..."):
                start_time = time.time()
                gamma_val = "scale" if gamma == "scale" else ("auto" if gamma == "auto" else gamma)
                svm = SVC(kernel=kernel, C=C_value, gamma=gamma_val, random_state=42)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                train_time = time.time() - start_time
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a: st.metric("Accuracy", f"{acc:.2%}")
                with col_b: st.metric("Precision", f"{prec:.2%}")
                with col_c: st.metric("Recall", f"{rec:.2%}")
                with col_d: st.metric("F1 Score", f"{f1:.2%}")
                st.info(f"⏱️ Training time: {train_time:.3f} seconds")

# ============================================
# TAB 4: Neural Network
# ============================================
with tab4:
    st.markdown("### 🧠 Neural Network (MLP)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        hidden_layers = st.selectbox("Hidden Layer Architecture", ["(64,)", "(64, 32)", "(128, 64)", "(64, 32, 16)"])
        activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])
        solver = st.selectbox("Optimizer", ["adam", "sgd"])
        max_iter = st.slider("Maximum Iterations", min_value=100, max_value=1000, value=500, step=50)
        run_btn_nn = st.button("🚀 Train Neural Network", type="primary", use_container_width=True)
    
    with col2:
        if run_btn_nn:
            with st.spinner("Training Neural Network..."):
                start_time = time.time()
                hidden_tuple = eval(hidden_layers)
                mlp = MLPClassifier(hidden_layer_sizes=hidden_tuple, activation=activation, solver=solver, max_iter=max_iter, random_state=42, early_stopping=True)
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)
                train_time = time.time() - start_time
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a: st.metric("Accuracy", f"{acc:.2%}")
                with col_b: st.metric("Precision", f"{prec:.2%}")
                with col_c: st.metric("Recall", f"{rec:.2%}")
                with col_d: st.metric("F1 Score", f"{f1:.2%}")
                st.info(f"⏱️ Training time: {train_time:.3f} seconds")
                
                # Loss curve
                st.markdown("---")
                st.markdown("### 📉 Training Loss Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=mlp.loss_curve_, mode='lines', name='Training Loss', line=dict(color='#1e3a5f', width=2)))
                fig.update_layout(title="Loss During Training", xaxis_title="Iteration", yaxis_title="Loss", height=300, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About Model Tuning")
    st.markdown("""
    **Hyperparameters** control how the model learns.
    
    | Parameter | Effect |
    |-----------|--------|
    | **k** (KNN) | Higher k = smoother boundary |
    | **max_depth** (DT) | Deeper = more complex |
    | **C** (SVM) | Higher C = more complex |
    | **hidden_layers** (NN) | More neurons = more capacity |
    
    ---
    ### 📊 Baseline Performance
    
    | Model | Accuracy |
    |-------|----------|
    | KNN (k=9) | 63.16% |
    | Decision Tree | 67.45% |
    | SVM-RBF | 68.87% |
    | **MLP** | **72.15%** |
    """)
    st.caption("Body Performance Analytics | Model Tuning Studio")
