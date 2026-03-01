import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

print("Loading dataset...")

# Load dataset (User must place Churn_Modelling.csv in this folder)
data = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns if present
drop_cols = ["RowNumber","CustomerId","Surname"]
for col in drop_cols:
    if col in data.columns:
        data = data.drop(col, axis=1)

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Target column (usually Exited = churn)
target = "Exited"

X = data.drop(target, axis=1)
y = data[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Gradient Boosting (strong for churn prediction)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl","wb"))

print("Model trained and saved successfully")
