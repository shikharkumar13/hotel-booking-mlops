import joblib                                                                                                                                          
import pandas as pd
from pathlib import Path                                                                                                                               
                                                                                                                                                    
MODEL_PATH = Path("models/random_forest.pkl")
                                                                                                                                                        
_pipeline = None
                                                                                                                                                        
                
def load_pipeline():                                                                                                                                   
    global _pipeline
    if _pipeline is None:                                                                                                                              
        _pipeline = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    return _pipeline                            
                    
                                                                                                                                                        
def engineer_features(data: dict) -> dict:
    data["total_guests"] = data["adults"] + data["children"] + data["babies"]                                                                          
    data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    return data                                                                                                                                        
                                                                                                                                                        
                                                                                                                                                        
def predict(booking: dict) -> dict:                                                                                                                    
    pipeline = load_pipeline()     
                            
    booking = engineer_features(booking.copy())
    df = pd.DataFrame([booking])                                                                                                                       
                                
    probability = pipeline.predict_proba(df)[0][1]                                                                                                     
    will_cancel = bool(probability >= 0.5)        
                                        
    if probability >= 0.7:                                                                                                                             
        confidence = "High"
    elif probability >= 0.4:                                                                                                                           
        confidence = "Medium"
    else:                    
        confidence = "Low"
                                                                                                                                                        
    return {
        "will_cancel": will_cancel,                                                                                                                    
        "cancellation_probability": round(float(probability), 4),
        "confidence": confidence                                 
    }