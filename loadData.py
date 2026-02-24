import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load the CSV
df = pd.read_csv('dataset.csv') # Replace with your actual filename

# 2. Extract X and y
# Assuming first column is 'disease' and the rest are symptoms
X = df.iloc[:, 1:].values  # Symptom matrix (377 columns)
y = df.iloc[:, 0].values   # Disease labels

# 3. Encode the diseases (e.g., "Pneumonia" -> 42)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 4. Split for testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Save the classes so you can decode them later
np.save('classes.npy', encoder.classes_)