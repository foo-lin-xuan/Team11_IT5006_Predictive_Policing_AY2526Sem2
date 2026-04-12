---
title: Crime Risk Prediction
emoji: ⚖️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# IT5006 Project Milestone 3: Integration & Communication - Deployment, Final Report & Presentation

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

## Models Deployed
- Logistic Regression as baseline
- Random Forest
- XGBoost as primary model
- Weighted Soft-Voting Ensemble (RF + XGBoost) 

## Milestone 3 Project Code
```
project-root
│   Dockerfile
│   README.md
├───deployment           
│    │   IT5006_Milestone3_Build_Pipelines.ipynb  
│    ├───api        # Web Service Deployment
│    └───ui         # Prediction UI
└────...

```

## Web Service Deployment

Deploy FastAPI service on [Hugging Face Spaces](https://huggingface.co/):
<!-- 
Note to include:
- Dockerfile
- YAML header in README.md 
-->

API URL: [https://foo-lin-xuan-crime-risk-prediction.hf.space/docs](https://foo-lin-xuan-crime-risk-prediction.hf.space/docs)

## Prediction UI

Deploy Streamlit application on [Streamlit Community Cloud](https://share.streamlit.io/).

Application URL: [https://crime-risk-prediction.streamlit.app/](https://crime-risk-prediction.streamlit.app/)