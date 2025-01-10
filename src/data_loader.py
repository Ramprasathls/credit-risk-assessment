import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load the dataset
    dataset = load_data(DATA_PATH)
    
    # Preprocess the dataset
    X_train, X_test, y_train, y_test = preprocess_data(dataset)
