import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. Synthetische Daten mit Logik generieren
def generate_data(n=2000):
    np.random.seed(42)
    data = {
        'price': np.random.uniform(5, 1000, n),           # Preis des Artikels
        'account_balance': np.random.uniform(100, 5000, n), # Verfügbares Guthaben
        'mood_score': np.random.randint(1, 11, n),        # 1 (Schlecht) bis 10 (Sehr gut)
        'is_limited_offer': np.random.choice([0, 1], n),  # FOMO-Faktor
        'sleep_hours': np.random.uniform(3, 10, n)        # Erholungsgrad
    }
    df = pd.DataFrame(data)
    
    # Logik für den Regret Score (0-100)
    # Höherer Stress, wenn der Preis einen großen Teil des Kontostands ausmacht
    financial_ratio = (df['price'] / df['account_balance']) * 40
    
    df['regret_score'] = (
        financial_ratio +                      # Finanzielle Belastung
        (11 - df['mood_score']) * 4 +          # Emotionale Verfassung
        (10 - df['sleep_hours']) * 2 +         # Körperliche Verfassung
        (df['is_limited_offer'] * 15)          # Zeitdruck
    )
    
    # Rauschen hinzufügen für Realismus und Begrenzung auf 0-100
    df['regret_score'] += np.random.normal(0, 5, n)
    df['regret_score'] = df['regret_score'].clip(0, 100)
    return df

# 2. Modell trainieren
print("Starte Offline-Training...")
df = generate_data()
X = df.drop('regret_score', axis=1)
y = df['regret_score']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Modell exportieren
if not os.path.exists('data'):
    os.makedirs('data')
joblib.dump(model, 'data/regret_model.pkl')
print("Erfolg: Modell gespeichert unter 'data/regret_model.pkl'")