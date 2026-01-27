import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os

# Create mlruns directory if it doesn't exist
if not os.path.exists("mlruns"):
    os.makedirs("mlruns")

def prepare_data(file_path="data/bank_churn.csv"):
    df = pd.read_csv(file_path)
    
    # Feature Selection: Drop irrelevant columns
    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = df['Exited']
    
    # Define categorical and numerical features
    categorical_features = ['Geography', 'Gender']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

def train_model(model_name="RandomForest", params=None):
    (X_train, X_test, y_train, y_test), preprocessor = prepare_data()
    
    # Set MLflow Experiment
    mlflow.set_experiment("Bank_Churn_Prediction")
    
    with mlflow.start_run(run_name=model_name):
        if model_name == "RandomForest":
            # Default params if none provided
            if params is None:
                params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
            model = RandomForestClassifier(**params)
        elif model_name == "XGBoost":
            if params is None:
                params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        
        # Build full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log Model
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"--- {model_name} Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Example: Run two experiments
    print("Running Random Forest Experiment...")
    train_model(model_name="RandomForest", params={"n_estimators": 150, "max_depth": 12})
    
    print("\nRunning XGBoost Experiment...")
    train_model(model_name="XGBoost", params={"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6})
    
    print("\nMLflow UI can be launched using: mlflow ui")
