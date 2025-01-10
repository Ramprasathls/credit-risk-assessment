import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Path to the dataset
DATA_PATH = r"E:\credit_risk_assessment\credit-risk-assessment\data\credit_risk_dataset.csv"

def load_data(file_path):
    """
    Load the dataset and display basic information.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Display dataset information
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nDataset Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    return data

def preprocess_data(data):
    """
    Preprocess the dataset:
    - Handle missing values
    - Encode categorical features
    - Normalize numerical features
    - Split data into train and test sets
    """
    # Step 1: Drop duplicates (if any)
    data = data.drop_duplicates()
    
    # Step 2: Handle missing values
    # Fill missing `person_emp_length` and `loan_int_rate` with their medians
    data.loc[:, 'person_emp_length'] = data['person_emp_length'].fillna(data['person_emp_length'].median())
    data.loc[:, 'loan_int_rate'] = data['loan_int_rate'].fillna(data['loan_int_rate'].median())
    
    # Step 3: Remove outliers
    # Cap `person_age` to remove unrealistic values (e.g., >100 years)
    data.loc[:, 'person_age'] = data['person_age'].apply(lambda x: min(x, 100))
    
    # Step 4: Separate features (X) and target (y)
    X = data.drop(columns=['loan_status'])  # `loan_status` is the target
    y = data['loan_status']
    
    # Step 5: Identify categorical and numerical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    print("\nNumerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)
    
    # Step 6: Define transformations
    # Standardize numerical features
    num_transformer = StandardScaler()
    
    # One-hot encode categorical features
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine transformations into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ]
    )
    
    # Step 7: Apply transformations
    # Fit and transform the features
    X_transformed = preprocessor.fit_transform(X)
    
    # Step 8: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    print("\nPreprocessing Completed!")
    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)
    
    return X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols

def train_model(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    print("\nTraining Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on the test data.
    """
    print("\nEvaluating Model...")
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def tune_logistic_regression(X_train, y_train):
    """
    Perform hyperparameter tuning for Logistic Regression using GridSearchCV.
    """
    print("\nTuning Logistic Regression...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)
    return grid_search.best_estimator_

def train_advanced_models(X_train, y_train):
    """
    Train advanced models like Random Forest and Gradient Boosting.
    """
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    print("Random Forest Training Completed!")
    
    print("\nTraining Gradient Boosting...")
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    print("Gradient Boosting Training Completed!")
    
    return rf_model, gb_model

def compare_models(models, X_test, y_test):
    """
    Compare the performance of different models on the test set.
    """
    print("\nComparing Models...")
    for name, model in models.items():
        print(f"\n{name}:")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

def save_model(model, preprocessor, model_path="random_forest_pipeline.pkl"):
    """
    Save the trained model and preprocessing pipeline as a single file.
    """
    print("\nSaving the model and preprocessor...")
    pipeline = {
        'model': model,
        'preprocessor': preprocessor
    }
    joblib.dump(pipeline, model_path)
    print(f"Model and pipeline saved to {model_path}")

def load_model(model_path="random_forest_pipeline.pkl"):
    """
    Load the saved model and preprocessing pipeline.
    """
    print("\nLoading the model and preprocessor...")
    pipeline = joblib.load(model_path)
    print("Model and pipeline loaded successfully!")
    return pipeline

def predict_new_data(pipeline, new_data):
    """
    Make predictions on new data using the saved pipeline.
    """
    print("\nMaking predictions on new data...")
    preprocessor = pipeline['preprocessor']
    model = pipeline['model']
    
    # Preprocess new data
    X_new = preprocessor.transform(new_data)
    
    # Make predictions
    predictions = model.predict(X_new)
    return predictions

# ---------------------- PREVIOUSLY EXECUTED CODE BLOCKS ----------------------
# 1. Preprocessing the data:
# Preprocessed the dataset to handle missing values, outliers, and class imbalances.
# Returned training and testing sets along with preprocessor, numerical_cols, and categorical_cols.

# 2. Evaluated the basic Random Forest model:
# Trained and evaluated a basic Random Forest model without addressing class imbalance.

# ---------------------------------------------------------------------------
#if __name__ == "__main__":
     # Load the dataset
#    dataset = load_data(DATA_PATH)
    
    # Preprocess the dataset
#    X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols = preprocess_data(dataset)
    
    # Tune Logistic Regression
    # best_lr_model = tune_logistic_regression(X_train, y_train)
    
    # Train advanced models
    # rf_model, gb_model = train_advanced_models(X_train, y_train)
    
    # Compare all models
    # models = {
    #    "Logistic Regression (Tuned)": best_lr_model,
    #    "Random Forest": rf_model,
    #    "Gradient Boosting": gb_model
    #}
    # compare_models(models, X_test, y_test)

    # Train the Random Forest model (skip other models for simplicity)
#    rf_model = RandomForestClassifier(random_state=42)
#   rf_model.fit(X_train, y_train)
    
    # Save the model and preprocessor
#    save_model(rf_model, preprocessor=preprocessor, model_path="random_forest_pipeline.pkl")
    
    # Evaluate the model
#    evaluate_model(rf_model, X_test, y_test)
    
    # Test the pipeline with a sample input
#    pipeline = load_model("random_forest_pipeline.pkl")
#    sample_data = pd.DataFrame([{
#        'person_age': 30,
#        'person_income': 50000,
#        'person_home_ownership': 'RENT',
#        'person_emp_length': 3.0,
#        'loan_intent': 'MEDICAL',
#        'loan_grade': 'C',
#        'loan_amnt': 15000,
#        'loan_int_rate': 12.5,
#        'loan_percent_income': 0.3,
#        'cb_person_default_on_file': 'N',
#        'cb_person_cred_hist_length': 6
#    }])
#   predictions = predict_new_data(pipeline, sample_data)
#    print("\nPredictions for new data:", predictions)

# Define the function to train a weighted Random Forest model
def train_weighted_random_forest(X_train, y_train):
    """
    Train a Random Forest model with class weighting to address class imbalance.
    """
    print("\nTraining Weighted Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')  # Adjusting class weights
    rf_model.fit(X_train, y_train)
    print("Weighted Random Forest Training Completed!")
    return rf_model

if __name__ == "__main__":
    # Load the dataset
    dataset = load_data(DATA_PATH)
    
    # Preprocess the dataset
    X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols = preprocess_data(dataset)
    
    # Train a weighted Random Forest model
    weighted_rf_model = train_weighted_random_forest(X_train, y_train)
    
    # Evaluate the weighted model
    evaluate_model(weighted_rf_model, X_test, y_test)
    
    # Save the weighted model
    save_model(weighted_rf_model, preprocessor, model_path="E:/credit_risk_assessment/credit-risk-assessment/models/weighted_random_forest_pipeline.pkl")
    
    # Test the pipeline with a sample input
    pipeline = load_model("E:/credit_risk_assessment/credit-risk-assessment/models/weighted_random_forest_pipeline.pkl")
    sample_data = pd.DataFrame([{
        'person_age': 30,
        'person_income': 50000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 3.0,
        'loan_intent': 'MEDICAL',
        'loan_grade': 'C',
        'loan_amnt': 15000,
        'loan_int_rate': 12.5,
        'loan_percent_income': 0.3,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 6
    }])
    predictions = predict_new_data(pipeline, sample_data)
    print("\nPredictions for new data:", predictions)