# ICU Length of Stay Prediction

This project aims to predict whether a patient's second ICU stay will be prolonged (≥ 3 days) using machine learning models. By combining structured data from the first ICU admission with unstructured discharge notes (transformed into embeddings), the model enhances prediction accuracy. The study builds on the work by Zhang and Kuo (2024), utilizing the MIMIC-IV dataset to train classifiers like Random Forest, XGBoost, and others. The project includes data preprocessing, feature engineering, model training, and the development of a Streamlit web app for interactive exploration.

## Overview

Accurately predicting patient outcomes in the intensive care unit (ICU) is crucial for hospital resource planning, patient care management, and overall healthcare system efficiency. Prolonged ICU stays are associated with higher complications, mortality rates, and costs. By predicting these outcomes early, hospitals can optimize bed usage, staffing, and resource allocation.

This project replicates and expands on a study by Zhang and Kuo (2024) by utilizing the MIMIC-IV dataset. It integrates medication history, vital signs, and diagnoses data from the first ICU admission and incorporates discharge note embeddings for more accurate predictions.

## Project Goals

- **Early Prediction**: Predict if a patient’s second ICU stay will be prolonged (≥ 3 days).
- **Data Fusion**: Combine structured data (demographics, medications, vitals) and unstructured data (discharge notes) for model enhancement.
- **Improved Accuracy**: Use advanced machine learning techniques (Random Forest, XGBoost, etc.) to optimize prediction.
- **Deployment**: Create a Streamlit application to interactively explore the data and predictions.

## Dataset

The dataset used in this project is the **MIMIC-IV** (Medical Information Mart for Intensive Care IV), a large de-identified clinical dataset developed by MIT. It contains data on over 65,000 ICU patients between 2008 and 2019, including admissions, diagnoses, medications, vital signs, lab results, and more.

## Methods

We used several machine learning models to predict the length of ICU stay, including:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- AdaBoost
- XGBoost

Additionally, discharge notes were processed using Natural Language Processing (NLP) techniques to generate embeddings that were integrated into the prediction models.

### Key Steps in the Pipeline:

1. **Data Extraction**: The data was filtered and merged from various MIMIC-IV tables to create a structured dataset using **SQLite** for efficient management.
2. **Text Processing**: Discharge notes were cleaned and tokenized, then transformed into vector embeddings using the **Doc2Vec** model.
3. **Model Training**: Various classifiers were trained and evaluated using cross-validation and hyperparameter tuning.
4. **Model Calibration**: Calibration techniques were applied to ensure accurate probability predictions.
5. **Feature Importance**: Key features influencing the prediction were identified, such as age, first admission length, and certain medications.

## SQLite Integration

In this project, **SQLite** was used for efficient data management and querying. The MIMIC-IV database tables were imported into a local SQLite instance, which allowed for easy filtering and merging of relevant data from multiple tables. This setup was integral for building the dataset that fed into the machine learning models.

- **SQLite Setup**: A local SQLite instance was used to store and manage MIMIC-IV data, ensuring that large datasets could be efficiently queried and processed.
- **SQL Queries**: Using Python’s `sqlite3` library, we executed SQL queries to extract and filter the necessary data from the MIMIC-IV tables, including admissions, diagnoses, prescriptions, and patient demographics.
- **Data Pipeline**: This database was used to extract features from both the first and second ICU admissions, which were then merged and processed for model training.

This SQLite-based data extraction pipeline ensures that the data preprocessing is reproducible and easy to maintain, allowing for future improvements and modifications.

## Results

- **Best Performing Models**: Random Forest and XGBoost achieved the highest AUC (Area Under the Curve) and F1-scores.
- **Improved Predictions**: Incorporating discharge note embeddings led to significant improvements in model recall and F1-score.

## Streamlit Application

A **Streamlit** web application was created to allow users to interactively explore the dataset and make predictions about patients' second ICU stays. The app connects to the MIMIC-IV database and provides a user-friendly interface for clinical decision support.

The key features of the Streamlit app include:
- **Data Exploration**: Query and filter MIMIC-IV data for specific patient records.
- **Prediction**: Generate predictions on whether a second ICU stay will be prolonged based on patient data from the first ICU admission.
- **Interactive Interface**: Users can dynamically explore various patient features and outcomes.

## Future Work

- **External Dataset Validation**: Evaluate model performance with external datasets from other hospitals to assess generalizability.
- **Deep Learning Models**: Incorporate time-series data and image signals for enhanced predictions.
- **Bias and Fairness**: Address fairness and bias issues related to age, gender, and ethnicity in ICU predictions.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ICU-Length-of-Stay-Prediction.git
   cd ICU-Length-of-Stay-Prediction
