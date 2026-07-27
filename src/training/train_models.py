import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# load processed data
df = pd.read_csv("data/processed/final.csv")

FEATURES = [
    'grid', 'qualifying_position', 
    'driver_points_before', 'driver_position_before', 'driver_wins_before', 
    'constructor_points_before', 'constructor_position_before', 'constructor_wins_before'
]
print(df[FEATURES].isnull().sum())
TARGET = 'podium'

train_df = df[df['year'] <= 2022]
test_df = df[df['year'] >= 2023]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
X_test = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)

print("Train: ", X_train.shape, " Test: ", X_test.shape)
print("train podium rate:", y_train.mean(), " test podium rate: ", y_test.mean())

# scale feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# smote on training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("train shape: ", X_train_resampled.shape)
print("podium rate: ", y_train_resampled.mean())

# train models
models = {
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test_scaled)

    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds)
    }

    print(f"\n {name}")
    for metric, value in results[name].items():
        print(metric, ":", value)

#save trained models
for name, model in models.items():
    filename = f"models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)

joblib.dump(scaler, "models/scaler.pkl")

print("saved")