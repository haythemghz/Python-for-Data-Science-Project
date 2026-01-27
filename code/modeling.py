import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

def prepare_data(file_path="data/bank_churn.csv"):
    df = pd.read_csv(file_path)
    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = df['Exited']
    
    categorical_features = ['Geography', 'Gender']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

def run_experiment(model_name, model, params=None):
    (X_train, X_test, y_train, y_test), preprocessor = prepare_data()
    
    mlflow.set_experiment("Bank_Churn_Ensembles")
    
    with mlflow.start_run(run_name=model_name):
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        if params:
            mlflow.log_params(params)
        
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(full_pipeline, "model")
        
        print(f"[{model_name}] Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    # 1. Simple Models
    print("Running Simple Models...")
    run_experiment("LogisticRegression", LogisticRegression())
    run_experiment("DecisionTree", DecisionTreeClassifier(max_depth=10))

    # 2. Advanced Boosting
    print("\nRunning Boosting...")
    run_experiment("XGBoost", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss'))

    # 3. Ensemble: Voting
    print("\nRunning Voting Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    xgb = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    voting_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
    run_experiment("Voting_RF_XGB", voting_model)

    # 4. Ensemble: Stacking
    print("\nRunning Stacking Classifier...")
    stacking_model = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        final_estimator=LogisticRegression()
    )
    run_experiment("Stacking_RF_XGB", stacking_model)

    print("\nAll experiments tracked in MLflow. Use 'mlflow ui' to compare.")
