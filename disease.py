import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

df = pd.read_csv("cleaned_merged_heart_dataset.csv")

print("Dataset Shape:", df.shape)

X = df.drop("target", axis=1)
y = df["target"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Support Vector Machine":
        SVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

results = {}

for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    if y_prob is not None:
        roc = roc_auc_score(y_test, y_prob)
        print("ROC-AUC  :", roc)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    results[name] = accuracy

best_model = max(results, key=results.get)

print("\nBest Model :", best_model)
print("Accuracy   :", results[best_model])

plt.figure(figsize=(8,5))

plt.bar(results.keys(), results.values())

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

final_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

final_model.fit(X_train, y_train)

sample = X_test[0].reshape(1,-1)

prediction = final_model.predict(sample)

print("\nPrediction for Sample Patient")

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")