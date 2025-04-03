import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Veri yükleme ve ön işleme
df = pd.read_csv(r'C:\GYK\project\processed_data.csv')
categorical_columns = ['product_name', 'category_name', 'yearquarter', 'city']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop(columns=['discount_effective', 'unit_price', 'order_date', 'customer_id', 'product_id', 'units_in_stock', 'category_id', 'Year'])
y = df['discount_effective']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE ile veri dengeleme
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Min-Max Scaling (KNN için gerekli)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# --- MODELLERİN EĞİTİMİ VE DEĞERLENDİRİLMESİ ---
results = []

# Decision Tree
dt_params = {"max_depth": [3, 5, 8, 10], "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 5, 10]}
dt_model = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=10, n_jobs=-1, verbose=0)
dt_model.fit(X_train_balanced, y_train_balanced)
best_dt = dt_model.best_estimator_
y_dt_pred = best_dt.predict(X_test)
y_dt_proba = best_dt.predict_proba(X_test)[:, 1]

results.append({
    "Model": "Decision Tree",
    "Accuracy": accuracy_score(y_test, y_dt_pred),
    "Precision": precision_score(y_test, y_dt_pred),
    "Recall": recall_score(y_test, y_dt_pred),
    "F1-Score": f1_score(y_test, y_dt_pred),
    "ROC-AUC": roc_auc_score(y_test, y_dt_proba)
})

# KNN
knn_params = {"n_neighbors": np.arange(1, 50)}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=10, n_jobs=-1, verbose=0)
knn_model.fit(X_train_scaled, y_train_balanced)
best_knn = knn_model.best_estimator_
y_knn_pred = best_knn.predict(X_test_scaled)
y_knn_proba = best_knn.predict_proba(X_test_scaled)[:, 1]

results.append({
    "Model": "KNN",
    "Accuracy": accuracy_score(y_test, y_knn_pred),
    "Precision": precision_score(y_test, y_knn_pred),
    "Recall": recall_score(y_test, y_knn_pred),
    "F1-Score": f1_score(y_test, y_knn_pred),
    "ROC-AUC": roc_auc_score(y_test, y_knn_proba)
})

# Logistic Regression
lr_params = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 200, 500]}
lr_model = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
lr_model.fit(X_train_balanced, y_train_balanced)
best_lr = lr_model.best_estimator_
y_lr_pred = best_lr.predict(X_test)
y_lr_proba = best_lr.predict_proba(X_test)[:, 1]

results.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_lr_pred),
    "Precision": precision_score(y_test, y_lr_pred),
    "Recall": recall_score(y_test, y_lr_pred),
    "F1-Score": f1_score(y_test, y_lr_pred),
    "ROC-AUC": roc_auc_score(y_test, y_lr_proba)
})

# --- SONUÇLARIN TABLOSU ---
results_df = pd.DataFrame(results)
print(results_df)

# --- ROC EĞRİLERİ ---
plt.figure(figsize=(10, 6))
for model_name, y_proba in zip(
    ["Decision Tree", "KNN", "Logistic Regression"],
    [y_dt_proba, y_knn_proba, y_lr_proba]
):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrileri")
plt.legend(loc="lower right")
plt.show()

# --- CONFUSION MATRIX GRAFİKLERİ ---
for model_name, y_pred in zip(
    ["Decision Tree", "KNN", "Logistic Regression"],
    [y_dt_pred, y_knn_pred, y_lr_pred]
):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()







































