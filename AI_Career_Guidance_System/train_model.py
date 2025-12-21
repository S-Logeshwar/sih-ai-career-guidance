import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

df = pd.read_csv('data/career_dataset.csv')
features = ['Database Fundamentals', 'Cloud Computing', 'Programming Skills', 
            'Hacking Skills', 'Management Skills', 'Game Development', 
            'Interest in AI', 'Math Score', 'Science Score', 'Personality Score']
X = df[features]
y = df['Career']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Accuracy:", model.score(X_test, y_test))
os.makedirs('model', exist_ok=True)
with open('model/career_model.pkl', 'wb') as f:
    pickle.dump((model, le), f)
print("Model trained and saved!")