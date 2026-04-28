from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.schemas import BookingRequest, BookingResponse, HealthResponse                                                                            
from src.models.predict import load_pipeline, predict                      
import logging                                                                                                                                         
                                                                                                                                                        
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)                                                                                                                   
                                    

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    load_pipeline()                                                                                                                                    
    logger.info("Model loaded. API is ready.")
    yield                                                                                                                                              
    logger.info("Shutting down.")
                                
                                                                                                                                                        
app = FastAPI(
    title="Hotel Booking Cancellation API",                                                                                                            
    description="Predicts the probability that a hotel booking will be cancelled.",
    version="1.0.0",                                                               
    lifespan=lifespan                                                                                                                                  
)                    
                                                                                                                                                        
                                                                                                                                                        
@app.get("/health", response_model=HealthResponse)
def health_check():                                                                                                                                    
    from src.models.predict import _pipeline
    return HealthResponse(status="ok", model_loaded=_pipeline is not None)
                                                                        
                                                                                                                                                        
@app.post("/predict", response_model=BookingResponse)
def predict_cancellation(booking: BookingRequest):                                                                                                     
    try:                                          
        result = predict(booking.model_dump())
        return BookingResponse(**result)                                                                                                               
    except Exception as e:              
        logger.error(f"Prediction error: {e}")                                                                                                         
        raise HTTPException(status_code=500, detail="Prediction failed.")