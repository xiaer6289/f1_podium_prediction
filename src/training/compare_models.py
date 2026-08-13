import pandas as pd
import joblib 
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/final.csv")

FEATURES = [
    'grid', 'qualifying_position', 
    'driver_points_before', 'driver_position_before', 'driver_wins_before', 
    'constructor_points_before', 'constructor_position_before', 'constructor_wins_before'
]
TARGET = 'podium'

test_df = df[df['year'] >= 2023]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)
X_test_scaled = scaler.transform(X_test_imputed)

MODEL_FILES = {
    "SVM (baseline)": "models/svm.pkl",
    "Random Forest (baseline)": "models/random_forest.pkl",
    "KNN (baseline)": "models/knn.pkl", 
    "SVM (tuned)": "models/svm_tuned.pkl",
    "Random Forest (tuned)": "models/random_forest_tuned.pkl",
    "KNN (tuned)": "models/knn_tuned.pkl"
}

records = []

for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"Skipping {name} - file not found: {path}")
        continue

    model = joblib.load(path)
    preds = model.predict(X_test_scaled)

    records.append({
        "Model": name, 
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds)
    })

    results_df = pd.DataFrame(records)
    print(results_df)

os.makedirs("reports", exist_ok=True)
results_df.to_csv("reports/model_comparison.csv", index=False)

metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = range(len(results_df))
bar_width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

for i, metric in enumerate(metrics):
    offsets = [pos + i * bar_width for pos in x]
    ax.bar(offsets, results_df[metric], width=bar_width, label=metric)

ax.set_xticks([pos + bar_width * 1.5 for pos in x])
ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Podium Prediction (Test Set 2023+)")
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("reports/model_comparison_chart.png", dpi=150)
print("Saved")
plt.show()