import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError
from src.api.schemas import BookingRequest

VALID_BOOKING = {
    "hotel": "Resort Hotel", "lead_time": 45,
    "arrival_date_month": "August", "arrival_date_week_number": 32,
    "arrival_date_day_of_month": 15, "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 5, "adults": 2, "children": 0, "babies": 0,
    "meal": "BB", "country": "PRT", "market_segment": "Online TA",
    "distribution_channel": "TA/TO", "is_repeated_guest": 0,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A", "assigned_room_type": "A",
    "booking_changes": 0, "deposit_type": "No Deposit",
    "days_in_waiting_list": 0, "customer_type": "Transient",
    "adr": 120.50, "required_car_parking_spaces": 0,
    "total_of_special_requests": 1
}


def test_valid_booking_accepted():
    booking = BookingRequest(**VALID_BOOKING)
    assert booking.hotel == "Resort Hotel"
    assert booking.lead_time == 45


def test_negative_lead_time_rejected():
    data = {**VALID_BOOKING, "lead_time": -1}
    with pytest.raises(ValidationError):
        BookingRequest(**data)


def test_missing_required_field_rejected():
    data = {k: v for k, v in VALID_BOOKING.items() if k != "hotel"}
    with pytest.raises(ValidationError):
        BookingRequest(**data)


def test_invalid_week_number_rejected():
    data = {**VALID_BOOKING, "arrival_date_week_number": 54}
    with pytest.raises(ValidationError):
        BookingRequest(**data)
