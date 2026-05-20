def create_features(data):

    X = data[["tenure", "monthly_charges", "total_charges"]]

    y = data["churn"]

    return X, y