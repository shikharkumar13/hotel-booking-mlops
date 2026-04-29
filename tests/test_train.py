import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline
from src.models.train import build_pipeline


def test_build_pipeline_returns_pipeline():
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    pipeline = build_pipeline(params)
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_has_preprocessor_and_model_steps():
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    pipeline = build_pipeline(params)
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "model" in step_names


def test_build_pipeline_step_order():
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    pipeline = build_pipeline(params)
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "model"
