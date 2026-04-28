from pydantic import BaseModel, Field                                                                                                                  
                                    
                                                                                                                                                        
class BookingRequest(BaseModel):                                                                                                                     
    hotel: str = Field(..., example="Resort Hotel")
    lead_time: int = Field(..., ge=0, example=45)  
    arrival_date_month: str = Field(..., example="August")                                                                                             
    arrival_date_week_number: int = Field(..., ge=1, le=53, example=32)
    arrival_date_day_of_month: int = Field(..., ge=1, le=31, example=15)                                                                               
    stays_in_weekend_nights: int = Field(..., ge=0, example=2)                                                                                       
    stays_in_week_nights: int = Field(..., ge=0, example=5)                                                                                            
    adults: int = Field(..., ge=0, example=2)                                                                                                        
    children: int = Field(..., ge=0, example=0)                                                                                                        
    babies: int = Field(..., ge=0, example=0)                                                                                                          
    meal: str = Field(..., example="BB")     
    country: str = Field(..., example="PRT")                                                                                                           
    market_segment: str = Field(..., example="Online TA")                                                                                              
    distribution_channel: str = Field(..., example="TA/TO")
    is_repeated_guest: int = Field(..., ge=0, le=1, example=0)                                                                                         
    previous_cancellations: int = Field(..., ge=0, example=0)                                                                                          
    previous_bookings_not_canceled: int = Field(..., ge=0, example=0)
    reserved_room_type: str = Field(..., example="A")                                                                                                  
    assigned_room_type: str = Field(..., example="A")                                                                                                
    booking_changes: int = Field(..., ge=0, example=0)                                                                                                 
    deposit_type: str = Field(..., example="No Deposit")                                                                                               
    days_in_waiting_list: int = Field(..., ge=0, example=0)
    customer_type: str = Field(..., example="Transient")                                                                                               
    adr: float = Field(..., ge=0, example=120.50)                                                                                                      
    required_car_parking_spaces: int = Field(..., ge=0, example=0)
    total_of_special_requests: int = Field(..., ge=0, example=1)                                                                                          
                                                                                                                                                        

class BookingResponse(BaseModel):                                                                                                                      
    will_cancel: bool                                                                                                                                
    cancellation_probability: float
    confidence: str
                                                                                                                                                        

class HealthResponse(BaseModel):                                                                                                                       
    status: str                                                                                                                                      
    model_loaded: bool