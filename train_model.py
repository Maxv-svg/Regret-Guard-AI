import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. Generate Advanced Synthetic Data
def generate_data(n=4000):
    np.random.seed(42)
    data = {
        'price': np.random.uniform(10, 1000, n),
        'account_balance': np.random.uniform(100, 5000, n),
        'mood_score': np.random.randint(1, 11, n),
        'is_limited_offer': np.random.choice([0, 1], n),
        'sleep_hours': np.random.uniform(3, 11, n),
        'merchant_risk_score': np.random.uniform(0.05, 0.50, n)
    }
    df = pd.DataFrame(data)
    
    # --- FEATURE ENGINEERING (Highest Grade Exploration) ---
    # Derived Feature 1: Relative Price (Financial Impact)
    df['relative_price'] = df['price'] / df['account_balance']
    
    # Derived Feature 2: Impulsivity Index (Psychological State)
    df['impulsivity_index'] = (
        (11 - df['mood_score']) * 0.4 + 
        (11 - df['sleep_hours']) * 0.4 + 
        (df['is_limited_offer'] * 2)
    )
    
    # Logic for Target: Ground Truth for Regret Score
    df['regret_score'] = (
        (df['relative_price'] * 100) + 
        (df['merchant_risk_score'] * 40) + 
        (df['impulsivity_index'] * 5)
    )
    df['regret_score'] = df['regret_score'].clip(0, 100) + np.random.normal(0, 2, n)
    return df

# 2. Pipeline and Evaluation (Accuracy Aspect)
df = generate_data()
X = df.drop('regret_score', axis=1)
y = df['regret_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Export Bundle with Metadata (Separation of Concerns)
if not os.path.exists('data'): os.makedirs('data')
bundle = {
    'model': model,
    'features': X.columns.tolist(),
    'mae': mean_absolute_error(y_test, model.predict(X_test))
}
joblib.dump(bundle, 'data/regret_bundle.pkl')
print(f"English Model Bundle Exported. MAE: {bundle['mae']:.2f}")