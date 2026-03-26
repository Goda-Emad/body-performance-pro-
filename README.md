# 🏋️ Body Performance Analytics and Intelligent Classification System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements a complete machine learning pipeline to analyze body performance data and predict:
- **Performance Class** (A=Best, B, C, D=Worst) using classification models
- **Broad Jump Distance** (explosive power) using regression models

The system provides an interactive web application with real-time predictions, model tuning, batch processing, and comprehensive reporting capabilities.

---

## 📊 Dataset

| Aspect | Details |
|--------|---------|
| **Source** | [Kaggle Body Performance Dataset](https://www.kaggle.com/datasets/kukuroo3/body-performance-data) |
| **Records** | 13,393 participants |
| **Features** | 12 attributes |
| **Target (Classification)** | class (A, B, C, D) |
| **Target (Regression)** | broad_jump_cm |
| **Class Balance** | Perfectly balanced (~3,348 per class) |

### Features Description

| Column | Description | Type |
|--------|-------------|------|
| age | Age of participant (years) | Numeric |
| gender | Biological sex (M/F) | Categorical |
| height_cm | Height in centimeters | Numeric |
| weight_kg | Body weight in kilograms | Numeric |
| body_fat_% | Body fat percentage | Numeric |
| diastolic | Diastolic blood pressure (mmHg) | Numeric |
| systolic | Systolic blood pressure (mmHg) | Numeric |
| gripForce | Grip strength measurement (kg) | Numeric |
| sit_and_bend_forward_cm | Flexibility measurement (cm) | Numeric |
| sit-ups counts | Muscular endurance (counts) | Numeric |
| broad_jump_cm | Explosive power (cm) | Numeric |
| class | Performance category (A/B/C/D) | Categorical |

---

## 🤖 Models Implemented

### Classification Models (4)
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **MLP (Neural Network)** | **72.15%** | **73.00%** | **72.15%** | **72.31%** |
| SVM-RBF | 68.87% | 69.68% | 68.87% | 69.04% |
| Decision Tree | 67.45% | 69.45% | 67.45% | 67.46% |
| KNN (k=9) | 63.16% | 65.10% | 63.16% | 63.52% |
| SVM-Linear | 61.14% | 61.13% | 61.14% | 61.03% |

### Regression Models (4)
| Model | R² Score | RMSE (cm) | MAE (cm) |
|-------|----------|-----------|----------|
| **MLP Regressor** | **0.7791** | **18.73** | **14.72** |
| SVR | 0.7749 | 18.90 | 14.89 |
| Linear Regression | 0.7658 | 19.28 | 15.12 |
| Decision Tree Regressor | 0.7221 | 21.00 | 16.45 |

---

## 🔬 Key Insights from EDA

| Insight | Correlation | Interpretation |
|---------|-------------|----------------|
| **Flexibility** | r = +0.59 | Strongest positive predictor |
| **Muscular Endurance** | r = +0.45 | Second strongest predictor |
| **Body Fat %** | r = -0.34 | Strongest negative predictor |
| **Explosive Power** | r = +0.26 | Moderate positive correlation |
| **Grip Strength** | r = +0.22 | Moderate positive correlation |

---

## 🛠️ Technical Stack

### Backend
- **Python 3.12** - Core programming language
- **scikit-learn 1.3.0** - Machine learning models and preprocessing
- **pandas 2.1.0** - Data manipulation and analysis
- **numpy 1.24.0** - Numerical computations

### Frontend
- **Streamlit 1.31.0** - Web application framework
- **Plotly 5.18.0** - Interactive visualizations
- **Matplotlib 3.7.0** - Static charts
- **Seaborn 0.12.0** - Statistical visualizations

### Reporting
- **reportlab 4.0.0** - PDF report generation
- **openpyxl 3.1.0** - Excel file support

---

## 📁 Project Structure
