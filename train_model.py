import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('Survey_Responses.csv', encoding='latin1')

# Preprocessing mappings
hour_map = {'0-2 hrs': 1, '2-5 hrs': 2, '5-8 hrs': 3, '8+ hrs': 4}
df['daily_hours_numeric'] = df['daily_hours'].map(hour_map)

# Encoding categorical variables
categorical_cols = ['fomo_feelings', 'social_comparision', 'morning_habit']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Define Features for Prediction
# We use habits to predict the Anxiety Score
features = ['daily_hours_numeric', 'check_frequency', 'fomo_feelings', 'social_comparision', 'morning_habit']
X = df[features]
y = df['anxiety_score']

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model and encoders
with open('anxiety_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model, 
        'encoders': encoders, 
        'hour_map': hour_map,
        'feature_names': features
    }, f)

print("anxiety_model.pkl created successfully!")