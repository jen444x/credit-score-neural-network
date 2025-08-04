# Credit Score Prediction Neural Network

Enhanced neural network architecture achieving **74.6% accuracy** on 80K+ customer credit records, improving from 60% baseline performance.

## 🎯 Project Overview

This project implements a deep learning solution for credit score classification, transforming raw customer financial data into accurate credit risk assessments. Starting from a basic neural network baseline, we optimized the architecture and preprocessing pipeline to achieve significant performance improvements.

## 📊 Performance Results

| Metric | Baseline | Improved | Gain |
|--------|----------|----------|------|
| **Accuracy** | 60.0% | **74.6%** | **+14.6%** |
| **Precision** | - | **72.8%** | - |
| **Recall** | - | **75.6%** | - |
| **F1-Score** | - | **73.9%** | - |

## 🚀 Key Improvements

### Architecture Enhancement
- **Optimized Network**: Redesigned from 24→48→96→96→48→3 to **32→64→128→128→64→3**
- **Feature Expansion**: Increased input features from 24 to 32 dimensions
- **Training Optimization**: Fine-tuned epochs and batch size for optimal convergence

### Advanced Data Pipeline
- **Sophisticated Preprocessing**: Implemented custom data cleaning functions
- **Feature Engineering**: Enhanced categorical encoding and numerical standardization
- **Missing Value Handling**: Intelligent imputation strategies for 80K+ records

### Comprehensive Evaluation
- **Multi-class Classification**: Handles Good/Poor/Standard credit scores
- **Confusion Matrix Analysis**: Detailed classification performance breakdown
- **Model Validation**: Robust train/test split with performance metrics

## 🛠️ Technical Implementation

### Dataset
- **Size**: 80,000 customer records
- **Features**: 29 original features → 32 engineered features
- **Target**: 3-class credit score classification (Good, Poor, Standard)

### Model Architecture
```python
Sequential([
    Dense(32, activation='relu', input_dim=32),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
```

### Technologies Used
- **Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab

## 📈 Model Performance

### Confusion Matrix Results
```
Predicted:    Good   Poor   Standard
Actual:
Good         2208     44      630
Poor          123   3658      907  
Standard     1018   1341     6071
```

### Class-wise Performance
- **Good Credit**: High precision, effective identification
- **Poor Credit**: Strong recall, good risk detection
- **Standard Credit**: Balanced performance across metrics

## 🎓 Academic Context

**Course**: CMPS3500 - Machine Learning  
**Team Project**: 4-person collaborative development  
**Objective**: Enhance baseline model performance through architecture optimization

## 📋 Requirements

```txt
tensorflow>=2.8.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 🔄 Usage

1. **Data Preparation**: Load and preprocess credit score dataset
2. **Model Training**: Execute neural network training pipeline  
3. **Evaluation**: Generate performance metrics and visualizations
4. **Prediction**: Apply trained model to new credit applications

## 🎯 Business Impact

This enhanced model provides:
- **Improved Risk Assessment**: 14.6% accuracy gain for better lending decisions
- **Reduced False Positives**: Higher precision minimizes good customer rejection
- **Enhanced Recall**: Better identification of high-risk applications
- **Scalable Solution**: Handles large-scale credit evaluation efficiently

---

*This project demonstrates practical machine learning optimization, transforming academic baseline code into a production-ready credit scoring solution.*
