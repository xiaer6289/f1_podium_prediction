# 🏎️ F1 Podium Prediction

A supervised machine learning application to predict whether a Formula 1 driver will finish on podium by using historical race data and three algorithm including SVM, Random Forest, and KNN to be compared

---

## 🧠 Overview

this machine learning model predict based on:

- Driver's Grid position
- Driver's Qualifying position
- Driver's championship points,
- Driver's standings
- Driver's wins 
- Constructor's championship
- Constructor's standing
- Constructor's wins

---

## ⚙️ Installation
 
### 1️⃣ Clone or download the project
 
```bash
git clone <your-repo-url>
cd f1_podium_prediction
```

### 2️⃣ Create a virtual environment
 
```bash
python -m venv .venv
```
 
Activate it:
 
**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 3️⃣ Install dependencies
 
```bash
pip install -r requirements.txt
```

### 4️⃣ Confirm the raw dataset is in place
 
Make sure these files exist under `data/raw/`:
 
```
constructor_standings.csv   constructors.csv        driver_standings.csv
drivers.csv                 races.csv                qualifying.csv
results.csv                 status.csv
```
 
📌 Dataset source: [Kaggle — Formula 1 Race Data by jtrotman](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data)
 
---
 
## 🔄 Updating the Dataset (New Race Data)
 
When new races happen and updated data is released, follow these steps to refresh predictions with the latest results:
 
### Step 1 — 📥 Download the updated CSVs
 
Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) and download the latest version.
 
### Step 2 — 🔁 Replace the raw files
 
Overwrite the existing files in `data/raw/` with the newly downloaded ones (same filenames — `races.csv`, `results.csv`, etc.).
 
### Step 3 — 🧹 Rebuild the processed dataset
 
Run the preprocessing script to regenerate `final.csv` with the new race data merged in:
 
```bash
python src/preprocessing/build_final_dataset.py
```

✅ You should see output confirming the new shape and podium class distribution, e.g.:
```
Saved final.csv with shape: (7200, 18)
podium
0    6180
1    1020
```
 
### Step 4 — 🏋️ Retrain the models
 
See [Retraining the Models](#️-retraining-the-models) below — this step is required, otherwise the app will keep using predictions based on the old data.
 
> ⚠️ **Important:** Skipping Step 4 means your GUI dropdowns will show new races, but predictions will still be generated using the *old* trained models — always retrain after updating data.
 
---
 
## 🏋️ Retraining the Models
 
### Quick baseline retrain (all 3 algorithms, default settings)
 
```bash
python src/training/train_models.py
```
 
### Full retrain with hyperparameter tuning (recommended)
 
```bash
python src/training/tune_rf.py
python src/training/tune_svm.py
python src/training/tune_knn.py
```
 
⏱️ **Note:** SVM tuning can take **8–15 minutes** depending on your machine — let it run without interrupting.
 
### Regenerate the comparison chart
 
```bash
python src/training/compare_models.py
```
 
This updates `reports/model_comparison_chart.png` and `reports/model_comparison.csv` with the latest results.
 
---
 
## 🖥️ Using the Application
 
### Launch the app

```bash
python src/gui/app.py
```

### How to make a prediction
 
| Step | Action |
|------|--------|
| 1️⃣ | Select a **Season** from the dropdown |
| 2️⃣ | Select a **Grand Prix** (updates based on season chosen) |
| 3️⃣ | Select a **Driver** (updates based on race chosen) |
| 4️⃣ | Select an **Algorithm** — Random Forest, SVM, KNN, or *Compare All* |
| 5️⃣ | Click **🔮 Predict** |
 
### Reading the result
 
```
Driver: Lewis Hamilton
Race: 2024 Monaco Grand Prix
 
Random Forest: Podium (84.0% confidence)
SVM: Podium (87.5% confidence)
KNN: Podium (79.2% confidence)
```
 
- **Prediction** — whether the model believes the driver finishes in the Top 3
- **Confidence %** — how certain the model is in that prediction
---
 
## 📊 Model Comparison
 
To view how the three algorithms perform against each other:
 
```bash
python src/training/compare_models.py
```

This generates:
- 📈 `reports/model_comparison_chart.png` — bar chart comparing Accuracy, Precision, Recall, F1
- 📄 `reports/model_comparison.csv` — raw metrics table for your report
---

## 📚 Requirements
 
Main packages used (see `requirements.txt` for exact pinned versions):
 
- 🐼 `pandas` — data loading & manipulation
- 🔢 `numpy` — numerical operations
- 🤖 `scikit-learn` — SVM, Random Forest, KNN, preprocessing, evaluation
- ⚖️ `imbalanced-learn` — SMOTE for class imbalance
- 💾 `joblib` — saving/loading trained models
- 📊 `matplotlib` — comparison charts
- 🖥️ `tkinter` — GUI *(ships with Python — no install needed)*