
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\yugig\OneDrive\Desktop\ML Project\dataset\water_potability.csv")

# Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Split before fitting preprocessing objects
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Impute missing values using training data only
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create output folders
os.makedirs("processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save processed datasets
pd.DataFrame(X_train, columns=X.columns).to_csv(
    "processed/X_train.csv", index=False
)
pd.DataFrame(X_test, columns=X.columns).to_csv(
    "processed/X_test.csv", index=False
)
y_train.to_csv("processed/y_train.csv", index=False)
y_test.to_csv("processed/y_test.csv", index=False)

# Save preprocessing objects for use in the app
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Preprocessing complete!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
