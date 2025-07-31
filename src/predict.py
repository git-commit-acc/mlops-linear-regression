import joblib
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import load_data
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def main():
    _, X_test, _, y_test = load_data()
    model = joblib.load("src/trained_model.joblib")
    preds = model.predict(X_test)
    print("Sample predictions:", preds[:5])
    print("Corresponding ground truths:", y_test[:5])

if __name__ == "__main__":
    main()
