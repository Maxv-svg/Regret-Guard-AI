import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. Advanced Synthetic Data Generation
def generate_data(n=3000):
    np.random.seed(42)
    data = {
        'price': np.random.uniform(10, 1000, n),
        'account_balance': np.random.uniform(100, 5000, n),
        'mood_score': np.random.randint(1, 11, n),
        'is_limited_offer': np.random.choice([0, 1], n),
        'sleep_hours': np.random.uniform(3, 11, n),
        'merchant_return_rate': np.random.uniform(0.02, 0.45, n)
    }
    df = pd.DataFrame(data)
    
    # Complex non-linear logic for Regret
    fin_pressure = (df['price'] / df['account_balance']) * 45
    df['regret_score'] = (
        fin_pressure + 
        (df['merchant_return_rate'] * 25) +
        (11 - df['mood_score']) * 4 + 
        (10 - df['sleep_hours']) * 2 +
        (df['is_limited_offer'] * 15)
    )
    df['regret_score'] = df['regret_score'].clip(0, 100) + np.random.normal(0, 4, n)
    return df

# 2. Pipeline with Evaluation
df = generate_data()
X = df.drop('regret_score', axis=1)
y = df['regret_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Export with metadata for the App
if not os.path.exists('data'): os.makedirs('data')
payload = {
    'model': model,
    'features': X.columns.tolist(),
    'mae': mean_absolute_error(y_test, model.predict(X_test))
}
joblib.dump(payload, 'data/regret_bundle.pkl')
print(f"Mind-blowing pipeline ready. MAE: {payload['mae']:.2f}")