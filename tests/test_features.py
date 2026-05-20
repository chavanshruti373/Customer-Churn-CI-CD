import os

from src.load_data import load_data
from src.feature_engineering import create_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "churn.csv")


def test_feature_creation():

    data = load_data(data_path)

    X, y = create_features(data)

    assert X is not None
    assert y is not None
    assert len(X) > 0