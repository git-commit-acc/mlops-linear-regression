from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=45):
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test