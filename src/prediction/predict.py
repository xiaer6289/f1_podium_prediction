# pyrefly: ignore [missing-import]
import joblib
import pandas as pd

FEATURES = [
    'grid', 'qualifying_position',
    'driver_points_before', 'driver_position_before', 'driver_wins_before',
    'constructor_points_before', 'constructor_position_before', 'constructor_wins_before'
]

MODEL_PATHS = {
    "Random Forest" : "models/random_forest_tuned.pkl",
    "SVM": "models/svm_tuned.pkl",
    "KNN": "models/knn_tuned.pkl"
}

# load scaler & imputer
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")

def load_model(name):
    return joblib.load(MODEL_PATHS[name])

def preprocess_input(input_dict):
    # eg. gird: 3, qualifying position:2 ...
    # return scaled, imputed array for prediction
    df = pd.DataFrame([input_dict], columns=FEATURES)
    df_imputed = pd.DataFrame(imputer.transform(df), columns=FEATURES)
    df_scaled = scaler.transform(df_imputed)
    return df_scaled

def predict_single(input_dict, model_name):
    model = load_model(model_name)
    X = preprocess_input(input_dict)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return prediction, probability

def predict_all(input_dict):
    results = {}
    for name in MODEL_PATHS:
        try:
            pred, prob = predict_single(input_dict, name)
            results[name] = {
                "prediction": "Podium" if pred == 1 else "Not Podium",
                "probability": round(prob * 100, 1)
            }
        except FileNotFoundError:
            results[name] = {"prediction": "Model not found", "Probability": None}
    return results

# testing
# if __name__ == "__main__":
#     sample_input = {
#         'grid': 1,
#         'qualifying_position': 1,
#         'driver_points_before': 200,
#         'driver_position_before': 1,
#         'driver_wins_before': 5,
#         'constructor_points_before': 350,
#         'constructor_position_before': 1,
#         'constructor_wins_before': 8
#     }

# results = predict_all(sample_input)

# for model_name, result in results.items():
#     print(f"{model_name}: {result['prediction']} ({result['probability']}% confidence)")