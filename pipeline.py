from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle # Changed from joblib
import os

def load_data():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    return X, y, housing.feature_names
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def save_artifacts(model, scaler, model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'house_price_model.pkl') # Changed extension
    scaler_path = os.path.join(model_dir, 'scaler.pkl') # Changed extension
    with open(model_path, 'wb') as f:
        pickle.dump(model, f) # Use pickle.dump
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f) # Use pickle.dump
    print(f"Model and scaler saved as .pkl files in '{model_dir}' directory.")

# Run the pipeline
if __name__ == "__main__":
    X, y, feature_names = load_data()
    print("Features:", feature_names)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    save_artifacts(model, scaler)