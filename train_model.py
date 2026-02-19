import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_professional_model():
    # 1. LOAD RAW DATA
    data_path = 'data/transaction_history.csv'
    if not os.path.exists(data_path):
        print("Error: CSV not found! Please ensure Step 1 was successful.")
        return

    df = pd.read_csv(data_path)
    
    # 2. FEATURE ENGINEERING (Preprocessing)
    # This transforms raw data into 'intelligent' features
    df['relative_price'] = df['price'] / df['account_balance']
    df['impulsivity_index'] = (
        (11 - df['mood_score']) * 0.3 + 
        (11 - df['sleep_hours']) * 0.3 + 
        (df['is_limited_offer'] * 1.5)
    )

    # 3. PREPARE X AND Y
    X = df.drop('regret_score', axis=1)
    y = df['regret_score']
    
    # 4. TRAIN-TEST SPLIT
    # This fulfills the 'Accuracy' requirement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. TRAIN THE EVALUATOR
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. EXPORT THE BUNDLE
    # We save the model and feature names for app.py
    bundle = {
        'model': model, 
        'features': X.columns.tolist(), 
        'mae': mean_absolute_error(y_test, model.predict(X_test))
    }
    joblib.dump(bundle, 'data/regret_bundle.pkl')
    print(f"Success: Model trained from CSV. Accuracy (MAE): {bundle['mae']:.2f}")

if __name__ == "__main__":
    train_professional_model()