import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
df = pd.read_csv('data/features_30_sec.csv')

# Choose fewer features
selected_features = ['tempo', 'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean']
X = df[selected_features]
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model and label encoder
joblib.dump(model, 'models/genre_classifier.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
print("\nModel and label encoder saved successfully!")
