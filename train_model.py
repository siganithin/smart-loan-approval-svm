import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

print("➡ Script started")

# ------------------ Load dataset ------------------
df = pd.read_csv("loan_data.csv")
print("✅ Dataset loaded")

# ------------------ Split X and y ------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})
print("✅ Target mapped")

# ------------------ Encoding ------------------
X = pd.get_dummies(X, drop_first=True)
print("✅ Encoding done")

# ------------------ Train-test split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✅ Train-test split done")

# ------------------ Handle missing values ------------------
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
print("✅ Missing values handled")

# ------------------ Scaling ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✅ Scaling done")

# ------------------ Train SVM models ------------------
svm_linear = SVC(kernel="linear")
svm_poly = SVC(kernel="poly", degree=3)
svm_rbf = SVC(kernel="rbf")

svm_linear.fit(X_train, y_train)
svm_poly.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)
print("✅ Models trained")

# ------------------ Save models ------------------
os.makedirs("models", exist_ok=True)

joblib.dump(svm_linear, "models/svm_linear.pkl")
joblib.dump(svm_poly, "models/svm_poly.pkl")
joblib.dump(svm_rbf, "models/svm_rbf.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(X.columns, "models/features.pkl")

print("🎉 ALL MODELS & FILES SAVED SUCCESSFULLY")
