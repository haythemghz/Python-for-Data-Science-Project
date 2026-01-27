import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_eda(file_path="data/bank_churn.csv"):
    """
    Performs Exploratory Data Analysis on the bank churn dataset.
    """
    df = pd.read_csv(file_path)
    
    print("--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Exited', data=df, palette='viridis')
    plt.title('Churn Count (Target Variable)')
    plt.savefig('data/churn_distribution.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Exited', y='Balance', data=df)
    plt.title('Balance vs Churn')
    plt.savefig('data/balance_vs_churn.png')
    plt.close()
    
    # Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('data/correlation_matrix.png')
    plt.close()
    
    print("\nEDA completed. Plots saved in 'data/' directory.")

if __name__ == "__main__":
    run_eda()
