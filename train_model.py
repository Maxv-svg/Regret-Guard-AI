import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. Advanced Synthetic Data Generation with Feature Engineering
def generate_data(n=4000):
    np.random.seed(42)
    data = {
        'price': np.random.uniform(10, 1000, n),
        'account_balance': np.random.uniform(100, 5000, n),
        'mood_score': np.random.randint(1, 11, n),
        'is_limited_offer': np.random.choice([0, 1], n),
        'sleep_hours': np.random.uniform(3, 11, n),
        'merchant_risk_score': np.random.uniform(0.05, 0.50, n) # Category/Merchant return history
    }
    df = pd.DataFrame(data)
    
    # --- FEATURE ENGINEERING ---
    # Feature 1: Relative Price (Impact on wallet)
    df['relative_price'] = df['price'] / df['account_balance']
    
    # Feature 2: Impulsivity Index (Psychological state)
    # Higher score = more impulsive (weighted by lack of sleep, low mood, and FOMO)
    df['impulsivity_index'] = (
        (11 - df['mood_score']) * 0.4 + 
        (11 - df['sleep_hours']) * 0.4 + 
        (df['is_limited_offer'] * 2)
    )
    
    # Target Logic: Ground Truth for Regret Score
    df['regret_score'] = (
        (df['relative_price'] * 100) + 
        (df['merchant_risk_score'] * 40) + 
        (df['impulsivity_index'] * 5)
    )
    
    df['regret_score'] = df['regret_score'].clip(0, 100) + np.random.normal(0, 2, n)
    return df

# 2. Pipeline and Accuracy Check
df = generate_data()
X = df.drop('regret_score', axis=1)
y = df['regret_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate Accuracy (MAE)
mae = mean_absolute_error(y_test, model.predict(X_test))

# 3. Export Bundle
if not os.path.exists('data'): os.makedirs('data')
bundle = {
    'model': model,
    'features': X.columns.tolist(),
    'mae': mae
}
joblib.dump(bundle, 'data/regret_bundle.pkl')
print(f"Model Bundle Exported. MAE: {mae:.2f}")