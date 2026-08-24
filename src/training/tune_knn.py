import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
import os

def tune_and_train_knn():
    print("Loading processed data...")
    # Go up two directories from src/training to reach data/processed
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "final.csv")
    df = pd.read_csv(data_path)
    print("\nFirst 10 rows of the loaded dataset:")
    print(df.head(10))

    FEATURES = [
        'grid', 'qualifying_position', 
        'driver_points_before', 'driver_position_before', 'driver_wins_before', 
        'constructor_points_before', 'constructor_position_before', 'constructor_wins_before'
    ]
    TARGET = 'podium'

    train_df = df[df['year'] <= 2022]
    test_df = df[df['year'] >= 2023]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print("\n" + "="*50)
    print("PHASE 1: Chronological Train-Test Split")
    print("="*50)
    print(f"Train set (2010-2022): {X_train.shape[0]:,} rows  |  Podium rate: {y_train.mean():.3f}")
    print(f"Test set  (2023-2025): {X_test.shape[0]:,} rows  |  Podium rate: {y_test.mean():.3f}")

    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"Train shape after SMOTE: {X_train_resampled.shape}")
    print(f"Podium rate after SMOTE: {y_train_resampled.mean():.3f}")

    print("\n" + "="*50)
    print("PHASE 2: 5-Fold Cross-Validation + Final Test Evaluation")
    print("="*50)
    k_values = range(1, 26)
    cv_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_resampled, y_train_resampled, cv=skf, scoring='f1')
        mean_score = scores.mean()
        cv_scores.append(mean_score)
        print(f"K={k}: Mean F1 = {mean_score:.4f}")

    best_idx = np.argmax(cv_scores)
    best_k = k_values[best_idx]
    best_f1 = cv_scores[best_idx]
    
    print(f"\nBest K found: {best_k} with F1 score: {best_f1:.4f}")

    print("\n--- Training Final KNN Model ---")
    # Using the same hyperparameters as in the original knn_f1_podium.py
    knn_final = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean', algorithm='ball_tree')
    knn_final.fit(X_train_resampled, y_train_resampled)

    print("\n--- Evaluating Model ---")
    preds = knn_final.predict(X_test_scaled)

    print("Accuracy :", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, zero_division=0))
    print("Recall   :", recall_score(y_test, preds, zero_division=0))
    print("F1 Score :", f1_score(y_test, preds, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    print("\n--- Saving Model ---")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "knn_tuned.pkl")
    joblib.dump(knn_final, model_path)
    print(f"KNN model successfully saved to {model_path}")

if __name__ == "__main__":
    tune_and_train_knn()
