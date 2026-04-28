import pandas as pd                                                                                                                                    
import os                                                                                                                                            
        
RAW_PATH = "data/raw/hotel_bookings.csv"
PROCESSED_PATH = "data/processed/hotel_features.csv"                                                                                                   
                                                    
                                                                                                                                                        
def load_and_clean(path: str) -> pd.DataFrame:                                                                                                         
    df = pd.read_csv(path)                    
                                                                                                                                                        
    # reservation_status reveals the answer directly — data leakage, must drop                                                                       
    # reservation_status_date is also derived from cancellation — drop it too                                                                          
    # agent and company have too many missing values to be useful            
    drop_cols = [                                                                                                                                      
        "reservation_status",                                                                                                                          
        "reservation_status_date",                                                                                                                     
        "agent",                                                                                                                                       
        "company"                                                                                                                                    
    ]                                                                                                                                                  
    df.drop(columns=drop_cols, inplace=True)                                                                                                         
                                            
    # Fill missing values
    df["children"].fillna(0, inplace=True)                                                                                                             
    df["country"].fillna("Unknown", inplace=True)
                                                                                                                                                        
    # Engineered features                                                                                                                              
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]                                                                    
                                                                                    
    return df                                                                                                                                          
                                                                                                                                                        
                                                                                                                                                        
def run():                                                                                                                                             
    os.makedirs("data/processed", exist_ok=True)
    df = load_and_clean(RAW_PATH)               
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_PATH}")
    print(f"Shape: {df.shape}")                                                                                                                        
    print(f"Columns: {list(df.columns)}")
                                                                                                                                                        
                                                                                                                                                        
if __name__ == "__main__":
    run()