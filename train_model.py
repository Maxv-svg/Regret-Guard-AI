import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. GENERATE DUMMY DATASET (The 'Evidence')
def create_csv_data(n=5000):
    np.random.seed(42)
    data = {
        'price': np.random.uniform(10, 1500, n),
        'account_balance': np.random.uniform(100, 6000, n),
        'mood_score': np.random.randint(1, 11, n),
        'is_limited_offer': np.random.choice([0, 1], n),
        'sleep_hours': np.random.uniform(3, 11, n),
        'merchant_risk_score': np.random.uniform(0.05, 0.55, n)
    }
    df = pd.DataFrame(data)
    
    # Feature Engineering (Logic the AI will learn)
    df['relative_price'] = df['price'] / df['account_balance']
    df['impulsivity_index'] = (
        (11 - df['mood_score']) * 0.3 + 
        (11 - df['sleep_hours']) * 0.3 + 
        (df['is_limited_offer'] * 1.5)
    )
    
    # Target Variable: Regret Score
    df['regret_score'] = (
        (df['relative_price'] * 80) + 
        (df['merchant_risk_score'] * 40) + 
        (df['impulsivity_index'] * 6)
    )
    df['regret_score'] = df['regret_score'].clip(0, 100) + np.random.normal(0, 2, n)
    
    # Save as CSV for audit purposes
    if not os.path.exists('data'): os.makedirs('data')
    df.to_csv('data/transaction_history.csv', index=False)
    print("✅ Step 1: Dummy data saved to 'data/transaction_history.csv'")

# 2. TRAIN MODEL FROM CSV
def train_from_csv():
    # Load the data we just created
    df = pd.read_csv('data/transaction_history.csv')
    
    X = df.drop('regret_score', axis=1)
    y = df['regret_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Accuracy Check
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    # Export Bundle
    bundle = {'model': model, 'features': X.columns.tolist(), 'mae': mae}
    joblib.dump(bundle, 'data/regret_bundle.pkl')
    print(f"✅ Step 2: Model trained from CSV. MAE: {mae:.2f}")

if __name__ == "__main__":
    create_csv_data()
    train_from_csv()