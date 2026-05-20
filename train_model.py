from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

import pandas as pd 

# Save background data
df = pd.DataFrame(X, columns=data.feature_names)

# Small representative sample
background = df.sample(50, random_state=42)

background.to_csv("background.csv", index=False)

print("Background data saved.")

# Save model
joblib.dump(model, "model.pkl")

print("Model saved.")