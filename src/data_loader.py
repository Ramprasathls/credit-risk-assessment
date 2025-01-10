import pandas as pd

# Path to the compressed dataset
DATA_PATH = "E:\credit_risk_assessment\credit-risk-assessment\data\credit_risk_dataset.csv"

def load_data(file_path):
    
    ## Load the dataset from the gzip file and display basic information.
    # Load the gzip file
    data = pd.read_csv(file_path)

    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(data.head())

    # Display summary information
    print("\nDataset Information:")
    print(data.info())

    # Display basic statistics
    print("\nDataset Statistics:")
    print(data.describe())

    return data

# Run the loader if the script is executed
if __name__ == "__main__":
    dataset = load_data(DATA_PATH)