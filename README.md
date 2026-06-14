# Predictive Maintenance — Engine Failure Classification

An end-to-end machine learning pipeline that predicts whether an engine requires maintenance or is operating normally, based on real-time sensor readings.

## Project Overview

This project is part of the PGP-AIML Capstone at Great Learning, University of Texas, Austin. It covers the complete ML lifecycle from data registration to automated deployment.

## Tech Stack

- **ML Models:** XGBoost, Random Forest (GridSearchCV tuning)
- **Experiment Tracking:** MLflow
- **Data & Model Registry:** Hugging Face Datasets and Model Hub
- **Deployment:** Streamlit on Hugging Face Spaces
- **Containerisation:** Docker
- **CI/CD:** GitHub Actions

## Live Application

https://huggingface.co/spaces/SaiSadhasivam/predictive-maintenance-app

## Dataset

https://huggingface.co/datasets/SaiSadhasivam/predictive-maintenance-capstone

## Model

https://huggingface.co/SaiSadhasivam/predictive-maintenance-best-model

## Pipeline

Every push to the main branch triggers the automated GitHub Actions workflow which runs data preparation, model training, model registration, and deployment end-to-end.

## Author

Sai Sadhasivam | PGP-AIML | University of Texas, Austin
