import pandas as pd
import joblib 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE 

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

joblib.dump(imputer, "models/imputer.pkl")

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

param_grid  ={
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring='f1', # optimize for F1 
    cv=5, # 5 fold cross validation
    n_jobs=-1,  # use all cpu core
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print('\nBest parameters: ', grid_search.best_params_)
print("Best CZV F1 score: ", grid_search.best_score_)

best_rf = grid_search.best_estimator_
preds = best_rf.predict(X_test_scaled)

print("\n random forest test set")
print("accuracy: ", accuracy_score(y_test, preds))
print("precision: ", precision_score(y_test, preds))
print("recall: ", recall_score(y_test, preds))
print("f1: ", f1_score(y_test, preds))
print("confusion matrix: \n", confusion_matrix(y_test, preds))

importances = pd.Series(best_rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("feature importances:\n ", importances)

# save
joblib.dump(best_rf, "models/random_forest_tuned.pkl")
print("\nsaved")
