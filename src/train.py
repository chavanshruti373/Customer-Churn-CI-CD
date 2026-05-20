import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.feature_engineering import create_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "churn.csv")


def train_model():

    data = load_data(data_path)

    X, y = create_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/churn_model.pkl")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()