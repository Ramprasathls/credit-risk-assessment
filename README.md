# Credit Risk Prediction System
A credit risk assessment application to predict defaulting of loans.

## Overview
The **Credit Risk Prediction System** is a machine learning-based application that predicts the likelihood of a loan applicant defaulting on their loan. This project demonstrates the entire data science workflow, from data preprocessing and model training to deployment using a web-based interface.

## Features
- Preprocesses and cleans the dataset to handle missing values, outliers, and class imbalance.
- Trains a **Random Forest Classifier** with class weighting for imbalanced data.
- Deploys a user-friendly **Streamlit web application** for real-time predictions.

## Dataset
- **Source**: [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Description**:
  - The dataset contains information about loan applicants, including demographics, financial details, loan information, and credit history.
  - Key columns include:
    - `person_age`: Age of the applicant.
    - `person_income`: Annual income of the applicant.
    - `loan_amnt`: Loan amount requested.
    - `loan_status`: Target variable (0 = No Default, 1 = Default).

## Workflow
1. **Data Preprocessing**:
   - Handled missing values using median imputation.
   - Scaled numerical features and encoded categorical features.
   - Capped outliers for realistic analysis.

2. **Model Development**:
   - Trained a **Random Forest Classifier** with class weighting to improve recall for the minority class (loan defaults).
   - Evaluated model performance using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

3. **Deployment**:
   - Built a **Streamlit web application** for real-time user interaction.
   - Users can input loan details and get predictions about default risk.

## Model Performance
- **Accuracy**: 93%
- **Precision for Defaults (Class 1)**: 97%
- **Recall for Defaults (Class 1)**: 72%
- **F1-Score for Defaults (Class 1)**: 83%

## Setup Instructions

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-risk-assessment.git
   cd credit-risk-assessment
