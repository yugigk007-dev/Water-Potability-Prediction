import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"C:\Users\yugig\OneDrive\Desktop\ML Project\dataset\water_potability.csv")
print(df.info())
print(df.isnull().sum())
df = df.drop_duplicates()
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)
print(df["Potability"].value_counts())
print("\nPercentage")
print(df["Potability"].value_counts(normalize=True) * 100)
df['Potability'].value_counts().plot(kind='bar')
plt.title("Potability Distribution")
plt.show()
plt.figure(figsize=(10,5))
plt.imshow(df.isnull(), aspect='auto')
plt.title("Missing Values")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()
for col in ["ph", "Sulfate", "Trihalomethanes"]:
    df[col] = df[col].fillna(df[col].median())
print(df.isnull().sum())
X = df.drop("Potability", axis=1)
y = df["Potability"]
print("Features Shape :", X.shape)
print("Target Shape :", y.shape)
X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.2,random_state=42,stratify=y)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:5])