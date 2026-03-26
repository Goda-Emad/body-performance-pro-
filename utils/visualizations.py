"""
Visualizations Module
---------------------
Creates interactive and static visualizations for the Body Performance Analytics application.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
import io
import base64


# Set style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['D', 'C', 'B', 'A'],
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Create an interactive confusion matrix using Plotly.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    class_names : list
        Names of classes
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Counts', 'Normalized (%)'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # Counts heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='Blues',
            showscale=True,
            name='Counts'
        ),
        row=1, col=1
    )
    
    # Normalized heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm_normalized * 100,
            x=class_names,
            y=class_names,
            text=cm_normalized.round(3) * 100,
            texttemplate='%{text:.1f}%',
            textfont={"size": 12},
            colorscale='Reds',
            showscale=True,
            name='Normalized'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        height=500,
        width=900,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="Actual Class")
    
    return fig


def create_comparison_chart(
    results: Dict[str, Dict[str, float]],
    metric: str = 'Accuracy',
    title: str = "Model Performance Comparison"
) -> go.Figure:
    """
    Create a bar chart comparing model performances.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model names and their metrics
    metric : str
        Metric to compare ('Accuracy', 'Precision', 'Recall', 'F1', 'RMSE', 'R2')
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]
    
    # Color based on value
    colors = []
    for v in values:
        if v >= 0.7:
            colors.append('#1e3a5f')  # Best
        elif v >= 0.65:
            colors.append('#2c4e6e')  # Good
        elif v >= 0.6:
            colors.append('#4682b4')  # Average
        else:
            colors.append('#94a3b8')  # Below average
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=values,
            text=[f"{v:.1%}" if v < 1 else f"{v:.2f}" for v in values],
            textposition='outside',
            marker_color=colors,
            hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric,
        height=500,
        template="plotly_white",
        showlegend=False
    )
    
    if metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        fig.update_yaxis(range=[0, 1], tickformat='.0%')
    
    return fig


def create_model_comparison_dashboard(
    classification_results: Dict[str, Dict[str, float]],
    regression_results: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Create a combined dashboard comparing all models.
    
    Parameters:
    -----------
    classification_results : dict
        Classification model results
    regression_results : dict
        Regression model results
    
    Returns:
    --------
    go.Figure
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Classification - Accuracy',
            'Classification - F1 Score',
            'Regression - R² Score',
            'Regression - RMSE (cm)'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Classification Accuracy
    clf_models = list(classification_results.keys())
    acc_values = [classification_results[m].get('Accuracy', 0) for m in clf_models]
    
    fig.add_trace(
        go.Bar(
            x=clf_models,
            y=acc_values,
            text=[f"{v:.1%}" for v in acc_values],
            textposition='outside',
            marker_color='#1e3a5f',
            name='Accuracy'
        ),
        row=1, col=1
    )
    
    # Classification F1
    f1_values = [classification_results[m].get('F1', 0) for m in clf_models]
    
    fig.add_trace(
        go.Bar(
            x=clf_models,
            y=f1_values,
            text=[f"{v:.1%}" for v in f1_values],
            textposition='outside',
            marker_color='#2c4e6e',
            name='F1 Score'
        ),
        row=1, col=2
    )
    
    # Regression R²
    reg_models = list(regression_results.keys())
    r2_values = [regression_results[m].get('R²', 0) for m in reg_models]
    
    fig.add_trace(
        go.Bar(
            x=reg_models,
            y=r2_values,
            text=[f"{v:.3f}" for v in r2_values],
            textposition='outside',
            marker_color='#10b981',
            name='R²'
        ),
        row=2, col=1
    )
    
    # Regression RMSE
    rmse_values = [regression_results[m].get('RMSE', 0) for m in reg_models]
    
    fig.add_trace(
        go.Bar(
            x=reg_models,
            y=rmse_values,
            text=[f"{v:.1f}" for v in rmse_values],
            textposition='outside',
            marker_color='#f59e0b',
            name='RMSE'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Model Performance Dashboard",
        height=800,
        width=1000,
        showlegend=False,
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Score", row=1, col=1, tickformat='.0%')
    fig.update_yaxes(title_text="Score", row=1, col=2, tickformat='.0%')
    fig.update_yaxes(title_text="R² Score", row=2, col=1)
    fig.update_yaxes(title_text="RMSE (cm)", row=2, col=2)
    
    return fig


def create_feature_importance_chart(
    feature_names: List[str],
    importances: List[float],
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Create a horizontal bar chart for feature importance.
    
    Parameters:
    -----------
    feature_names : list
        Feature names
    importances : list
        Importance scores
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imps = [importances[i] for i in sorted_idx]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_imps,
            y=sorted_names,
            orientation='h',
            marker_color='#1e3a5f',
            text=[f"{v:.3f}" for v in sorted_imps],
            textposition='outside',
            hovertemplate='%{y}: %{x:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_prediction_gauge(
    value: float,
    min_val: float = 0,
    max_val: float = 100,
    title: str = "Confidence Score"
) -> go.Figure:
    """
    Create a gauge chart for prediction confidence.
    
    Parameters:
    -----------
    value : float
        Confidence value
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Determine color
    if value >= 0.8:
        color = '#10b981'  # Green
    elif value >= 0.6:
        color = '#f59e0b'  # Orange
    else:
        color = '#ef4444'  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': '#fee2e2'},
                {'range': [60, 80], 'color': '#fed7aa'},
                {'range': [80, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


def create_distribution_plot(
    data: pd.Series,
    feature_name: str,
    bins: int = 50
) -> go.Figure:
    """
    Create a distribution plot with histogram and KDE.
    
    Parameters:
    -----------
    data : pd.Series
        Data to plot
    feature_name : str
        Name of the feature
    bins : int
        Number of bins
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        name='Histogram',
        marker_color='#1e3a5f',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # KDE
    from scipy import stats
    kde = stats.gaussian_kde(data.dropna())
    x_range = np.linspace(data.min(), data.max(), 200)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde(x_range),
        name='Density',
        line=dict(color='#f59e0b', width=2),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)'
    ))
    
    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.1f}")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_val:.1f}")
    
    fig.update_layout(
        title=f"Distribution of {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Density",
        height=400,
        template="plotly_white"
    )
    
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Heatmap"
) -> go.Figure:
    """
    Create a correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} ↔ %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        width=800,
        template="plotly_white"
    )
    
    return fig


def create_scatter_colored(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = None
) -> go.Figure:
    """
    Create a colored scatter plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    x_col : str
        X-axis column
    y_col : str
        Y-axis column
    color_col : str
        Column for coloring
    title : str
        Chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title or f"{x_col} vs {y_col} by {color_col}",
        opacity=0.6,
        color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981', '#1e3a5f']
    )
    
    fig.update_layout(
        height=500,
        template="plotly_white"
    )
    
    return fig


def fig_to_base64(fig: go.Figure) -> str:
    """
    Convert Plotly figure to base64 string for embedding in HTML/PDF.
    
    Parameters:
    -----------
    fig : go.Figure
        Plotly figure
    
    Returns:
    --------
    str
        Base64 encoded image
    """
    img_bytes = fig.to_image(format="png", width=800, height=500)
    base64_str = base64.b64encode(img_bytes).decode()
    return base64_str


def create_static_chart(
    data: pd.Series,
    chart_type: str = 'histogram',
    title: str = None
) -> io.BytesIO:
    """
    Create a static matplotlib chart for PDF reports.
    
    Parameters:
    -----------
    data : pd.Series
        Data to plot
    chart_type : str
        Type of chart ('histogram', 'boxplot', 'bar')
    title : str
        Chart title
    
    Returns:
    --------
    io.BytesIO
        Image buffer
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if chart_type == 'histogram':
        ax.hist(data, bins=50, color='#1e3a5f', alpha=0.7, edgecolor='white')
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
        ax.axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.1f}')
        ax.legend()
    
    elif chart_type == 'boxplot':
        ax.boxplot(data.dropna(), patch_artist=True,
                   boxprops=dict(facecolor='#1e3a5f', color='#1e3a5f'),
                   medianprops=dict(color='red', linewidth=2))
    
    elif chart_type == 'bar':
        counts = data.value_counts()
        ax.bar(counts.index, counts.values, color='#1e3a5f')
        ax.set_ylabel('Count')
    
    ax.set_title(title or f"Distribution of {data.name}")
    ax.set_xlabel(data.name)
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf


if __name__ == "__main__":
    # Test the module
    print("Testing visualizations module...")
    
    # Test correlation heatmap
    test_df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    })
    fig = create_correlation_heatmap(test_df)
    print("✅ Correlation heatmap created")
    
    # Test comparison chart
    test_results = {
        'Model A': {'Accuracy': 0.72, 'F1': 0.71},
        'Model B': {'Accuracy': 0.68, 'F1': 0.67},
        'Model C': {'Accuracy': 0.65, 'F1': 0.64}
    }
    fig = create_comparison_chart(test_results, 'Accuracy')
    print("✅ Comparison chart created")
    
    print("\n✅ visualizations.py loaded successfully")
