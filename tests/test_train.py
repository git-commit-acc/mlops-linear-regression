import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

def test_model_creation():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

def test_model_training():
    X_train, X_test, y_train, y_test = load_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, 'coef_')
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    assert r2 > 0.5  
