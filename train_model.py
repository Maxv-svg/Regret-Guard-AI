import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_model_pipeline():
    # 1. LOAD RAW DATA
    # We assume 'transaction_history.csv' contains the base raw columns
    data_path = 'data/transaction_history.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Ensure the CSV is in the data folder.")
        return

    df = pd.read_csv(data_path)
    print(f"Raw data loaded ({len(df)} rows). Starting Feature Engineering...")

    # 2. FEATURE ENGINEERING (Inside the Pipeline)
    # Relative Price: Measures the financial impact on the user's specific wallet
    df['relative_price'] = df['price'] / df['account_balance']
    
    # Impulsivity Index: Combines psychological and external FOMO factors
    df['impulsivity_index'] = (
        (11 - df['mood_score']) * 0.3 + 
        (11 - df['sleep_hours']) * 0.3 + 
        (df['is_limited_offer'] * 1.5)
    )

    # 3. PREPARE FEATURES AND TARGET
    # X contains our raw features + our new engineered features
    X = df.drop('regret_score', axis=1)
    y = df['regret_score']
    
    # 4. SPLIT FOR EVALUATION
    # Splitting allows us to calculate the 'Accuracy' of our prototype
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. TRAIN (Offline Logic)
    # Using a Random Forest to capture the non-linear relationship between features
    print("Training the AI brain...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. PERFORMANCE CHECK
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    # 7. EXPORT THE BUNDLE
    # We save everything the app needs to function in real-time
    bundle = {
        'model': model, 
        'features': X.columns.tolist(), 
        'mae': mae
    }
    
    if not os.path.exists('data'): 
        os.makedirs('data')
        
    joblib.dump(bundle, 'data/regret_bundle.pkl')
    print(f"Success: Model trained with engineered features. MAE: {mae:.2f}")

if __name__ == "__main__":
    train_model_pipeline()