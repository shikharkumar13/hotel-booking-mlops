import pandas as pd
import joblib                                                                                                                                          
import yaml
import os                                                                                                                                              
import mlflow                                                                                                                                        
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split                                                                                                   
from sklearn.metrics import (
    classification_report,                                                                                                                             
    roc_auc_score,                                                                                                                                   
    accuracy_score,                                                                                                                                    
    f1_score,
    precision_score,                                                                                                                                   
    recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder                                                                                                       

PROCESSED_PATH = "data/processed/hotel_features.csv"                                                                                                   
PARAMS_PATH = "params.yaml"
MODEL_PATH = "models/random_forest.pkl"                                                                                                                
EXPERIMENT_NAME = "hotel-booking-classifier"
                                                                                                                                                        
CATEGORICAL_FEATURES = [
    "hotel",                                                                                                                                           
    "arrival_date_month",
    "meal",
    "country",                                                                                                                                         
    "market_segment",
    "distribution_channel",                                                                                                                            
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type"                                                                                                                                    
]
                                                                                                                                                        
NUMERICAL_FEATURES = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",                                                                                                                       
    "stays_in_weekend_nights",
    "stays_in_week_nights",                                                                                                                            
    "adults",   
    "children",                                                                                                                                        
    "babies",   
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",                                                                                                                  
    "booking_changes",
    "days_in_waiting_list",                                                                                                                            
    "adr",                                                                                                                                             
    "required_car_parking_spaces",
    "total_of_special_requests",                                                                                                                       
    "total_guests",
    "total_nights"                                                                                                                                     
]                                                                                                                                                      

                                                                                                                                                        
def build_pipeline(params: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",                                                                                                    
                unknown_value=-1
            ), CATEGORICAL_FEATURES),                                                                                                                  
            ("num", "passthrough", NUMERICAL_FEATURES)
        ]                                                                                                                                              
    )
    return Pipeline([                                                                                                                                  
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=params["n_estimators"],                                                                                                       
            max_depth=params["max_depth"],
            random_state=params["random_state"],                                                                                                       
            n_jobs=-1
        ))                                                                                                                                             
    ])
                                                                                                                                                        
                
def run():
    os.makedirs("models", exist_ok=True)

    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)["model"]
                                                                                                                                                        
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)                                                                                                              
    mlflow.set_experiment(EXPERIMENT_NAME)                                                                                                             

    df = pd.read_csv(PROCESSED_PATH)                                                                                                                   
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df["is_canceled"]                                                                                                                              

    X_train, X_test, y_train, y_test = train_test_split(                                                                                               
        X, y,   
        test_size=params["test_size"],                                                                                                                 
        random_state=params["random_state"]
    )                                                                                                                                                  
                
    print(f"Training with params: {params}")                                                                                                           
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
                                                                                                                                                        
    with mlflow.start_run():                                                                                                                           

        mlflow.log_params(params)                                                                                                                      
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))                                                                                                     

        pipeline = build_pipeline(params)                                                                                                              
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)                                                                                                              
        y_proba = pipeline.predict_proba(X_test)[:, 1]
                                                                                                                                                        
        metrics = {
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "accuracy":  accuracy_score(y_test, y_pred),                                                                                               
            "f1":        f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),                                                                                              
            "recall":    recall_score(y_test, y_pred),
        }                                                                                                                                              
        mlflow.log_metrics(metrics)

        print("\n--- Evaluation ---")                                                                                                                  
        for name, value in metrics.items():
            print(f"{name:10s}: {value:.4f}")                                                                                                          
        print("\nFull Report:")                                                                                                                        
        print(classification_report(y_test, y_pred))
                                                                                                                                                        
        mlflow.sklearn.log_model(
            sk_model=pipeline,                                                                                                                         
            artifact_path="model"
        )                                                                                                                                              

        joblib.dump(pipeline, MODEL_PATH)                                                                                                              
        print(f"\nModel saved locally to {MODEL_PATH}")
        print("Run logged to MLflow.")
                                                                                                                                                        

if __name__ == "__main__":                                                                                                                             
    run()