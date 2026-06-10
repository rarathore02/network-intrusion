# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)

# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv("data/dataset.csv")

# ==========================================
# 3. Feature Selection
# ==========================================

selected_cols = [
    "Pkt Size Avg",
    "Protocol",
    "Flow Duration",
    "Flow Byts/s",
    "Flow Pkts/s"
]

# Keep selected features + label
df = df[selected_cols + ["Label"]]

# Convert labels:
# Benign = 0
# Bot = 1
df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "Benign" else 1
)

# ==========================================
# 4. Data Cleaning & Preprocessing
# ==========================================

# Replace infinity values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values using median
for col in selected_cols:
    df[col] = df[col].fillna(df[col].median())

# ==========================================
# 5. Prepare Features & Labels
# ==========================================

X = df[selected_cols]
y = df["Label"]

# ==========================================
# 6. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 7. Model Training
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 8. Predictions & Evaluation
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== Model Results ==========")
print("Accuracy Score:", accuracy)
print("F1 Score:", f1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 9. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Bot"]
)

display.plot()
plt.title("Confusion Matrix")

# Save image
plt.savefig("images/confusion_matrix.png")
plt.show()

# ==========================================
# 10. ROC Curve
# ==========================================

y_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("\nAUC Score:", roc_auc)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Save image
plt.savefig("images/roc_curve.png")
plt.show()

# ==========================================
# 11. Feature Importance
# ==========================================

importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(selected_cols, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")

# Save image
plt.savefig("images/feature_importance.png")
plt.show()