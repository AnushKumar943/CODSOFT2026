import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading Titanic Dataset...")
df = pd.read_csv("Titanic-Dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors='ignore')

# Fill missing values (Fixed Warning Version)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical features
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 🔥 Manual Prediction Part
# =========================

print("\nEnter Passenger Details to Predict Survival:")

pclass = int(input("Passenger Class (1/2/3): "))
sex = input("Sex (male/female): ").lower()
age = float(input("Age: "))
sibsp = int(input("Number of Siblings/Spouses aboard: "))
parch = int(input("Number of Parents/Children aboard: "))
fare = float(input("Fare: "))
embarked = input("Embarked (C/Q/S): ").upper()

# Encode input
sex = le_sex.transform([sex])[0]
embarked = le_embarked.transform([embarked])[0]

# Create input array
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)[0]

print("\nPrediction Result:")
if prediction == 1:
    print("The passenger SURVIVED 🎉")
else:
    print("The passenger DID NOT SURVIVE 💀")
