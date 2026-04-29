import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import engineer_features


def test_engineer_features_total_guests():
    booking = {
        "adults": 2, "children": 1, "babies": 0,
        "stays_in_weekend_nights": 2, "stays_in_week_nights": 3
    }
    result = engineer_features(booking.copy())
    assert result["total_guests"] == 3


def test_engineer_features_total_nights():
    booking = {
        "adults": 2, "children": 0, "babies": 0,
        "stays_in_weekend_nights": 1, "stays_in_week_nights": 4
    }
    result = engineer_features(booking.copy())
    assert result["total_nights"] == 5


def test_engineer_features_zero_children_and_babies():
    booking = {
        "adults": 1, "children": 0, "babies": 0,
        "stays_in_weekend_nights": 0, "stays_in_week_nights": 2
    }
    result = engineer_features(booking.copy())
    assert result["total_guests"] == 1
    assert result["total_nights"] == 2
