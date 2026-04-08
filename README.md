---
title: Crime Risk Prediction
emoji: ⚖️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# IT5006 Project Milestone 2: Analytics Implementation - Model Building & Evaluation

## Group
Group 11  
Yizhuo Zhang, Lin Xuan Foo, Yiding Cui, Yinan Jin, Xinyao Tan

## Overview
This project implements and evaluates multiple machine learning models to predict crime hotspots in Chicago.  
Using Chicago Crime Data from 2018 to 2025, we perform data preprocessing, feature engineering, model training, threshold tuning, time-aware validation, and explainability analysis.

The goal is to identify high-risk spatial-temporal crime units that can support data-driven police resource allocation.

## Dataset
- Source: Chicago Data Portal
- Time range: 2018-2025
- Raw dataset size: 1,952,048 records
- Aggregated dataset size after preprocessing: 1,478,884 records

## Methods
The notebook includes the following stages:
- Data loading and initial exploration
- Data cleaning and preprocessing
- Feature engineering
- Correlation analysis
- Model building and evaluation
- Explainability analysis with SHAP
- Practical interpretation and deployment recommendations

## Models Implemented
- Logistic Regression as baseline
- Random Forest
- XGBoost as primary model
- LSTM for sequential temporal modeling

## Evaluation Strategy
We used:
- Train / threshold / test split
- Time-aware validation
- Hyperparameter tuning
- Threshold optimization
- Standard classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC-ROC

## Key Results
Among the tested models, XGBoost achieved the best overall performance.

### Logistic Regression
- Accuracy: 0.6981
- Precision: 0.3710
- Recall: 0.6245
- F1-score: 0.4654
- AUC-ROC: 0.7361

### Random Forest
- Accuracy: 0.9229
- Precision: 0.9714
- Recall: 0.6529
- F1-score: 0.7809
- AUC-ROC: 0.9336

### XGBoost
- Accuracy: 0.9618
- Precision: 0.9913
- Recall: 0.8259
- F1-score: 0.9011
- AUC-ROC: 0.9475

In addition, the notebook includes an LSTM benchmark to compare sequential deep learning performance against tabular models.

## Explainability
SHAP analysis was applied to the XGBoost model to interpret prediction drivers.

Main findings include:
- Temporal features are highly important
- Crime type indicators contribute strongly to prediction
- Historical rolling features help capture hotspot persistence
- Spatial features help distinguish neighborhood-level risk patterns

## Business / Policy Implications
The results suggest that predictive policing resources can be improved through:
- Dynamic patrol scheduling by time of day
- Targeted deployment by crime type
- Early warning based on rolling historical crime intensity
- Community-specific strategies by spatial profile

## File Structure
- `IT5006_Milestone2_PredictivePolicing_Chicago_ModelBuilding_Evaluation.ipynb`: main implementation notebook

## Requirements
Recommended Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow
- shap
- scipy
See requirements.txt

## How to Run
1. Open the notebook in Jupyter Notebook or JupyterLab.
2. Install the required dependencies. Via `pip install -r requirements.txt`
3. Make sure the Chicago crime dataset is available locally or update the data-loading path in the notebook.
4. Run the notebook from top to bottom.

## Notes
- XGBoost is treated as the main production-ready model in this milestone.
- The LSTM section is included mainly for comparison with tabular methods.
- SHAP is used to support interpretation for the final report.

## Conclusion
This milestone demonstrates a complete analytics pipeline for crime hotspot prediction, from preprocessing and feature engineering to model evaluation and interpretation.  
The final results show that XGBoost provides the strongest predictive performance and the best balance between accuracy, interpretability, and practical deployment value.
