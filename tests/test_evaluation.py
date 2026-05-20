import os

from src.evaluate import evaluate_model


def test_model_evaluation():

    precision, recall = evaluate_model()

    assert precision >= 0.0
    assert recall >= 0.0

    assert os.path.exists("models/churn_model.pkl")