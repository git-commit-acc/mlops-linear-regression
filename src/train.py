from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import load_data

def main():
    X_train, X_test, y_train, y_test = load_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

    joblib.dump(model, "src/trained_model.joblib")

if __name__ == "__main__":
    main()