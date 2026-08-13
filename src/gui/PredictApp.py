import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import sys
import os
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'prediction'))
# pyrefly: ignore [missing-import]
from predict import predict_all, FEATURES

# Try loading initial df if it exists
try:
    df = pd.read_csv("data/processed/final.csv")
except Exception:
    df = pd.DataFrame()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data"

class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("F1 Podium Predictor")
        self.root.geometry("900x700")
        
        self.app_state = {
            'df': None,
            'features_df': None,
            'X_train_scaled': None,
            'X_test_scaled': None,
            'y_train_res': None,
            'y_test': None
        }

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.tab_processing = ttk.Frame(self.notebook)
        self.tab_prediction = ttk.Frame(self.notebook)
        self.tab_evaluation = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_processing, text="Data Processing")
        self.notebook.add(self.tab_evaluation, text="Model Evaluation")
        self.notebook.add(self.tab_prediction, text="Prediction Model")

        self.build_processing_widgets()
        self.build_widgets(self.tab_prediction)
        self.build_evaluation_widgets(self.tab_evaluation)

    def build_processing_widgets(self):
        left_frame = tk.Frame(self.tab_processing, width=220, bg="#f8f9fa")
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        right_frame = tk.Frame(self.tab_processing)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        tk.Label(left_frame, text="Main Menu", font=("Segoe UI", 14, "bold"), bg="#f8f9fa", fg="#2C3E50").pack(pady=15)

        button_style = {"width": 30, "font": ("Segoe UI", 9), "pady": 5}
        
        tk.Button(left_frame, text="Show Processed Dataset", command=self.show_dataset_head, bg="#D9EDF7", **button_style).pack(pady=(0, 15))
        
        tk.Button(left_frame, text="1. Data Loading & Initial Merge", command=self.stage_1, **button_style).pack(pady=5)
        tk.Button(left_frame, text="2. Target Variable Creation", command=self.stage_2, **button_style).pack(pady=5)
        tk.Button(left_frame, text="3. Feature Engineering", command=self.stage_3, **button_style).pack(pady=5)
        tk.Button(left_frame, text="4. Missing Value Handling", command=self.stage_4, **button_style).pack(pady=5)
        tk.Button(left_frame, text="5. Split + SMOTE + Scaling", command=self.stage_5, **button_style).pack(pady=5)
        tk.Button(left_frame, text="6. View Model Results (Prediction)", command=lambda: self.notebook.select(self.tab_prediction), bg="#D4EDDA", **button_style).pack(pady=(15, 5))

        self.output_text = scrolledtext.ScrolledText(right_frame, wrap=tk.NONE, bg="white", borderwidth=0, padx=10, pady=10)
        self.output_text.pack(fill="both", expand=True)
        
        x_scroll = tk.Scrollbar(right_frame, orient="horizontal", command=self.output_text.xview)
        x_scroll.pack(side="bottom", fill="x")
        self.output_text.configure(xscrollcommand=x_scroll.set)
        
        # Configure tags for beautiful formatting
        self.output_text.tag_config("header", font=("Segoe UI", 16, "bold"), foreground="#2C3E50", spacing1=15, spacing3=5)
        self.output_text.tag_config("subheader", font=("Segoe UI", 12, "bold"), foreground="#2980B9", spacing1=10, spacing3=2)
        self.output_text.tag_config("normal", font=("Segoe UI", 10), foreground="#333333", spacing1=2, spacing3=2)
        self.output_text.tag_config("mono", font=("Consolas", 10), foreground="#333333", spacing1=2, spacing3=2)
        self.output_text.tag_config("dataframe", font=("Consolas", 9), foreground="#2C3E50", background="#F1F3F5", spacing1=5, spacing3=5)
        self.output_text.tag_config("error", font=("Segoe UI", 10, "bold"), foreground="#C0392B", spacing1=5, spacing3=5)
        
        self.output_text.config(state=tk.DISABLED)
        self.show_dataset_head()

    def _insert_text(self, text, tag):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"{text}\n", tag)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

    def log_header(self, text):
        self._insert_text(text, "header")

    def log_subheader(self, text):
        self._insert_text(text, "subheader")

    def log_text(self, text):
        self._insert_text(text, "normal")
        
    def log_mono(self, text):
        self._insert_text(text, "mono")

    def log_error(self, text):
        self._insert_text(text, "error")

    def log_df(self, data):
        if isinstance(data, pd.Series):
            df_str = data.to_frame().to_string()
        else:
            df_str = data.to_string()
        self._insert_text(df_str, "dataframe")

    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    def show_dataset_head(self):
        self.clear_output()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        self.log_header("Dataset loaded from data/processed/final.csv")
        
        if df.empty:
            self.log_error("Dataset is empty or could not be loaded.")
        else:
            self.log_text(f"Total Shape: {df.shape}")
            self.log_subheader("First 10 rows:")
            self.log_df(df.head(10))
            
    def stage_1(self):
        self.clear_output()
        self.log_header("STAGE 1: Data Loading & Initial Merge")
        
        files = {
            'results': 'results.csv',
            'races': 'races.csv',
            'qualifying': 'qualifying.csv',
            'driver_standings': 'driver_standings.csv',
            'constructor_standings': 'constructor_standings.csv',
            'status': 'status.csv'
        }
        
        dfs = {}
        self.log_subheader("Original CSV counts:")
        for name, file in files.items():
            path = os.path.join(DATA_DIR, file)
            try:
                dfs[name] = pd.read_csv(path)
                self.log_mono(f"• {file:25} : {len(dfs[name]):,} rows")
            except FileNotFoundError:
                self.log_error(f"• {file:25} : ERROR - File not found at {path}")
                return
                
        # Merging
        merged_df = dfs['results'].merge(dfs['races'], on='raceId', suffixes=('', '_race'))
        merged_df = merged_df.merge(dfs['qualifying'], on=['raceId', 'driverId'], suffixes=('', '_qual'), how='left')
        merged_df = merged_df.merge(dfs['driver_standings'], on=['raceId', 'driverId'], suffixes=('', '_driver'), how='left')
        merged_df = merged_df.merge(dfs['constructor_standings'], on=['raceId', 'constructorId'], suffixes=('', '_const'), how='left')
        merged_df = merged_df.merge(dfs['status'], on='statusId', suffixes=('', '_stat'), how='left')
        
        merged_df = merged_df[(merged_df['year'] >= 2010) & (merged_df['year'] <= 2023)].copy()
        
        useful_cols = ['raceId', 'driverId', 'constructorId', 'year', 'round', 'name', 'grid', 'position_qual', 
                       'position_driver', 'points_driver', 'position_const', 'points_const', 'status', 'position', 'wins']
        useful_cols = [c for c in useful_cols if c in merged_df.columns]
        
        self.log_subheader("First 5 rows (Merged & Filtered):")
        self.log_df(merged_df[useful_cols].head())
        self.log_text(f"Shape after this stage: {merged_df[useful_cols].shape[0]:,} rows, {merged_df[useful_cols].shape[1]} columns")
        
        self.app_state['df'] = merged_df

    def stage_2(self):
        self.clear_output()
        self.log_header("STAGE 2: Target Variable Creation")
        
        local_df = self.app_state.get('df')
        if local_df is None:
            self.log_error("Please run Stage 1 first!")
            return
            
        local_df['position'] = pd.to_numeric(local_df['position'], errors='coerce')
        local_df['Podium'] = ((local_df['position'] <= 3) & (local_df['status'] == 'Finished')).astype(int)
        
        self.log_subheader("Class distribution:")
        counts = local_df['Podium'].value_counts()
        for val, count in counts.items():
            class_name = "Podium" if val == 1 else "No Podium"
            self.log_mono(f"• {class_name:10} (Class {val}) : {count:,} rows")
        
        useful_cols = ['raceId', 'driverId', 'year', 'name', 'position', 'status', 'Podium']
        useful_cols = [c for c in useful_cols if c in local_df.columns]
        
        self.log_subheader("First 5 rows (with new target):")
        self.log_df(local_df[useful_cols].head())
        self.log_text(f"Shape after this stage: {local_df.shape[0]:,} rows, {local_df.shape[1]} columns")
        self.app_state['df'] = local_df

    def stage_3(self):
        self.clear_output()
        self.log_header("STAGE 3: Feature Engineering")
        
        local_df = self.app_state.get('df')
        if local_df is None or 'Podium' not in local_df.columns:
            self.log_error("Please run Stage 1 and 2 first!")
            return
            
        STREET_CIRCUITS = ["monaco", "singapore", "baku", "miami", "las vegas", "albert park", "marina bay", "jeddah"]
        
        features = pd.DataFrame()
        features['grid_position'] = pd.to_numeric(local_df['grid'], errors='coerce')
        features['qualifying_position'] = pd.to_numeric(local_df['position_qual'], errors='coerce')
        features['driver_champ_position'] = pd.to_numeric(local_df['position_driver'], errors='coerce')
        features['driver_champ_points'] = pd.to_numeric(local_df['points_driver'], errors='coerce')
        features['constructor_champ_position'] = pd.to_numeric(local_df['position_const'], errors='coerce')
        features['constructor_champ_points'] = pd.to_numeric(local_df['points_const'], errors='coerce')
        features['race_round'] = pd.to_numeric(local_df['round'], errors='coerce')
        features['season_year'] = pd.to_numeric(local_df['year'], errors='coerce')
        
        circuit_names = local_df['name_circuit'].str.lower() if 'name_circuit' in local_df.columns else local_df['name'].str.lower()
        features['is_street_circuit'] = circuit_names.apply(
            lambda x: 1 if isinstance(x, str) and any(s in x for s in STREET_CIRCUITS) else 0
        )
        features['driver_wins_season'] = pd.to_numeric(local_df['wins'], errors='coerce')
        features['finished_race'] = (local_df['status'] == 'Finished').astype(int)
        features['Podium'] = local_df['Podium']
        
        self.log_subheader("First 5 rows (Features + Target):")
        self.log_df(features.head())
        self.log_text(f"Shape after this stage: {features.shape[0]:,} rows, {features.shape[1]} columns")
        self.app_state['features_df'] = features

    def stage_4(self):
        self.clear_output()
        self.log_header("STAGE 4: Missing Value Handling & Preprocessing")
        
        features = self.app_state.get('features_df')
        if features is None:
            self.log_error("Please run Stage 3 first!")
            return
            
        self.log_subheader("Missing values before filling:")
        self.log_df(features.isnull().sum())
        
        features['qualifying_position'] = features['qualifying_position'].fillna(features['grid_position'])
        
        medians = features.median()
        features['driver_champ_position'] = features['driver_champ_position'].fillna(medians['driver_champ_position'])
        features['constructor_champ_position'] = features['constructor_champ_position'].fillna(medians['constructor_champ_position'])
        
        features['driver_champ_points'] = features['driver_champ_points'].fillna(0)
        features['constructor_champ_points'] = features['constructor_champ_points'].fillna(0)
        features['driver_wins_season'] = features['driver_wins_season'].fillna(0)
        
        features = features.fillna(medians)
        
        self.log_subheader("First 5 rows after filling:")
        self.log_df(features.head())
        self.log_text(f"Missing values remaining: {features.isnull().sum().sum()}")
        self.log_text(f"Shape after this stage: {features.shape[0]:,} rows, {features.shape[1]} columns")
        
        self.app_state['features_df'] = features

    def stage_5(self):
        self.clear_output()
        self.log_header("STAGE 5: Train-Test Split + SMOTE + Scaling")
        
        features = self.app_state.get('features_df')
        if features is None or features.isnull().sum().sum() > 0:
            self.log_error("Please run Stage 4 first (Ensure no missing values)!")
            return
            
        train_mask = features['season_year'] <= 2021
        test_mask = features['season_year'] >= 2022
        
        X = features.drop(columns=['Podium', 'season_year'])
        y = features['Podium']
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        self.log_text(f"Train set size: {X_train.shape[0]:,} rows")
        self.log_text(f"Test set size: {X_test.shape[0]:,} rows")
        
        self.log_subheader("Class distribution before SMOTE (Train):")
        for val, count in y_train.value_counts().items():
            class_name = "Podium" if val == 1 else "No Podium"
            self.log_mono(f"• {class_name:10} (Class {val}) : {count:,} rows")
        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        self.log_subheader("Class distribution after SMOTE (Train):")
        for val, count in y_train_res.value_counts().items():
            class_name = "Podium" if val == 1 else "No Podium"
            self.log_mono(f"• {class_name:10} (Class {val}) : {count:,} rows")
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        self.log_subheader("First 5 rows of scaled training features:")
        self.log_df(X_train_scaled_df.head())
        self.log_text(f"Shape after this stage: Train features {X_train_scaled.shape}")
        
        self.app_state['X_train_scaled'] = X_train_scaled
        self.app_state['X_test_scaled'] = X_test_scaled
        self.app_state['y_train_res'] = y_train_res
        self.app_state['y_test'] = y_test

    def build_widgets(self, parent):
        # Main container
        main_frame = tk.Frame(parent, bg="#f8f9fa")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        tk.Label(main_frame, text="🏎️ Race Prediction Engine", font=("Segoe UI", 18, "bold"), bg="#f8f9fa", fg="#2C3E50").pack(pady=(0, 20))
        
        # Controls Frame (Card style)
        controls_frame = tk.Frame(main_frame, bg="white", padx=30, pady=20, highlightbackground="#dee2e6", highlightthickness=1)
        controls_frame.pack(fill="x", padx=80)

        # Helper to create styled labels
        def create_label(text):
            tk.Label(controls_frame, text=text, font=("Segoe UI", 10, "bold"), bg="white", fg="#495057").pack(pady=(15, 2))

        create_label("Season")
        self.season_var = tk.StringVar()
        seasons = sorted(df['year'].unique().tolist(), reverse=True) if not df.empty else []
        self.season_dropdown = ttk.Combobox(controls_frame, textvariable=self.season_var, values=seasons, state="readonly", font=("Segoe UI", 10), width=40)
        self.season_dropdown.pack()
        self.season_dropdown.bind("<<ComboboxSelected>>", self.on_season_selected)

        create_label("Grand Prix")
        self.race_var = tk.StringVar()
        self.race_dropdown = ttk.Combobox(controls_frame, textvariable=self.race_var, values=[], state="readonly", font=("Segoe UI", 10), width=40)
        self.race_dropdown.pack()
        self.race_dropdown.bind("<<ComboboxSelected>>", self.on_race_selected)

        create_label("Driver")
        self.driver_var = tk.StringVar()
        self.driver_dropdown = ttk.Combobox(controls_frame, textvariable=self.driver_var, values=[], state="readonly", font=("Segoe UI", 10), width=40)
        self.driver_dropdown.pack()

        create_label("Algorithm")
        self.algo_var = tk.StringVar()
        self.algo_dropdown = ttk.Combobox(controls_frame, textvariable=self.algo_var, 
                                        values=["Random Forest", "SVM", "KNN", "Compare All"], state="readonly", font=("Segoe UI", 10), width=40)
        self.algo_dropdown.current(3)
        self.algo_dropdown.pack(pady=(0, 10))

        # Predict Button
        tk.Button(main_frame, text="RUN PREDICTION", command=self.on_predict, bg="#E10600", fg="white", 
                  font=("Segoe UI", 12, "bold"), activebackground="#B30500", activeforeground="white",
                  relief="flat", cursor="hand2", padx=40, pady=10).pack(pady=25)

        # Results Frame (Terminal style)
        results_frame = tk.Frame(main_frame, bg="#2C3E50", padx=20, pady=20)
        results_frame.pack(fill="x", padx=80)
        
        self.result_text = tk.Text(results_frame, height=8, bg="#2C3E50", fg="#ECF0F1", 
                                   font=("Consolas", 11), borderwidth=0, highlightthickness=0)
        self.result_text.pack(fill="both", expand=True)
        self.result_text.insert("1.0", "Awaiting input...\n")
        
        # Tags for coloring the prediction output
        self.result_text.tag_config("info", foreground="#BDC3C7")
        self.result_text.tag_config("model", foreground="#3498DB", font=("Consolas", 11, "bold"))
        self.result_text.tag_config("prediction", foreground="#2ECC71", font=("Consolas", 11, "bold"))
        self.result_text.tag_config("nopodium", foreground="#E74C3C", font=("Consolas", 11, "bold"))
        self.result_text.config(state="disabled")

    def on_season_selected(self, event=None):
        if df.empty: return
        selected_year = int(self.season_var.get())
        races_in_season = df[df['year'] == selected_year]['name'].unique().tolist()
        self.race_dropdown['values'] = races_in_season
        self.race_var.set("")
        self.driver_dropdown['values'] = []
        self.driver_var.set("")

    def on_race_selected(self, event=None):
        if df.empty: return
        selected_year = int(self.season_var.get())
        selected_race = self.race_var.get()
        subset = df[(df['year'] == selected_year) & (df['name'] == selected_race)]
        drivers = (subset['forename'] + " " + subset['surname']).unique().tolist()
        self.driver_dropdown['values'] = drivers
        self.driver_var.set("")

    def on_predict(self):
        if df.empty:
            messagebox.showerror("Error", "Data is not loaded correctly.")
            return

        if not (self.season_var.get() and self.race_var.get() and self.driver_var.get()):
            messagebox.showwarning("Missing Selection", "Please select season, race, and driver")
            return 

        selected_year = int(self.season_var.get())
        selected_race = self.race_var.get()
        selected_driver = self.driver_var.get()

        subset = df[(df['year'] == selected_year) & (df['name'] == selected_race)]
        subset = subset[(subset['forename'] + " " + subset['surname']) == selected_driver]

        if subset.empty:
            messagebox.showerror("Error", "No data found for this selection")
            return
            
        row = subset.iloc[0]
        input_dict = {feat: row[feat] for feat in FEATURES}

        results = predict_all(input_dict)
        selected_algo = self.algo_var.get()
        
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        
        self.result_text.insert(tk.END, f"Driver: ", "info")
        self.result_text.insert(tk.END, f"{selected_driver}\n")
        self.result_text.insert(tk.END, f"Race:   ", "info")
        self.result_text.insert(tk.END, f"{selected_year} {selected_race}\n\n")
        
        for model_name, result in results.items():
            if selected_algo != "Compare All" and selected_algo != model_name:
                continue
            conf = result['probability']
            conf_str = f"({conf}% confidence)" if conf is not None else ""
            pred = result['prediction']
            pred_tag = "prediction" if pred == "Podium" else "nopodium"
            
            self.result_text.insert(tk.END, f"{model_name:14}: ", "model")
            self.result_text.insert(tk.END, f"{pred:10}", pred_tag)
            self.result_text.insert(tk.END, f"{conf_str}\n", "info")
            
        self.result_text.config(state="disabled")

    def build_evaluation_widgets(self, parent):
        main_frame = tk.Frame(parent, bg="#f8f9fa")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        tk.Label(main_frame, text="📊 Model Evaluation", font=("Segoe UI", 18, "bold"), bg="#f8f9fa", fg="#2C3E50").pack(pady=(0, 20))
        
        controls_frame = tk.Frame(main_frame, bg="#f8f9fa")
        controls_frame.pack(pady=10)
        
        tk.Label(controls_frame, text="Select Model:", font=("Segoe UI", 10, "bold"), bg="#f8f9fa").pack(side="left", padx=5)
        self.eval_model_var = tk.StringVar()
        self.eval_model_dropdown = ttk.Combobox(controls_frame, textvariable=self.eval_model_var, 
                                                values=["Random Forest", "SVM", "KNN", "Compare Model"], state="readonly", width=20)
        self.eval_model_dropdown.current(0)
        self.eval_model_dropdown.pack(side="left", padx=5)
        
        self.eval_canvas_frame = tk.Frame(main_frame, bg="white", highlightbackground="#dee2e6", highlightthickness=1)
        self.eval_canvas_frame.pack(expand=True, fill="both", padx=40, pady=10)
        
        tk.Button(main_frame, text="Generate Confusion Matrix", command=self.generate_confusion_matrix, bg="#3498DB", fg="white", 
                  font=("Segoe UI", 12, "bold"), activebackground="#2980B9", activeforeground="white",
                  relief="flat", cursor="hand2", padx=40, pady=10).pack(pady=20)

    def generate_confusion_matrix(self):
        if df.empty:
            messagebox.showerror("Error", "Data is not loaded correctly.")
            return

        try:
            for widget in self.eval_canvas_frame.winfo_children():
                widget.destroy()

            # pyrefly: ignore [missing-import]
            import joblib
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            # pyrefly: ignore [missing-import]
            import matplotlib.pyplot as plt
            # pyrefly: ignore [missing-import]
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import os
            
            temp_df = df.copy()
            if 'year' in temp_df.columns:
                temp_df = temp_df[temp_df['year'] >= 2023]
            
            # final.csv has a 'podium' column
            target_col = 'podium' if 'podium' in temp_df.columns else 'Podium'

            # pyrefly: ignore [missing-import]
            from predict import FEATURES
            
            # Drop rows only if the target is missing (features will be imputed)
            subset = temp_df.dropna(subset=[target_col])
            
            selected_model = self.eval_model_var.get()
            
            # Paths to models
            base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
            
            model_files = {
                "KNN": "knn.pkl",
                "Random Forest": "random_forest.pkl",
                "SVM": "svm.pkl"
            }
            
            if selected_model == "Compare Model":
                import subprocess
                import sys
                script_path = os.path.join(base_dir, 'src', 'training', 'compare_models.py')
                
                # Check if the process is already running
                if hasattr(self, 'compare_process') and self.compare_process.poll() is None:
                    messagebox.showinfo("Info", "The Compare Model window is already open. Please check your taskbar!")
                    return
                    
                self.compare_process = subprocess.Popen([sys.executable, script_path])
                return
            
            model_path = os.path.join(base_dir, 'models', model_files[selected_model])
            imputer_path = os.path.join(base_dir, 'models', 'imputer.pkl')
            scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
            
            model = joblib.load(model_path)
            imputer = joblib.load(imputer_path)
            scaler = joblib.load(scaler_path)
            
            X = subset[FEATURES]
            y_true = subset[target_col]
            
            X_imp = imputer.transform(X)
            X_scaled = scaler.transform(X_imp)
            
            y_pred = model.predict(X_scaled)
            
            cm = confusion_matrix(y_true, y_pred)
            
            if selected_model == "KNN" or selected_model == "Random Forest" or selected_model == "SVM":
                plt.close('all')  # Close previous figures to prevent them from getting stuck
                fig, ax = plt.subplots(figsize=(6, 5))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Podium", "Podium"])
                disp.plot(cmap="Blues", ax=ax)
                ax.set_title(f"{selected_model} Confusion Matrix")
                
                canvas = FigureCanvasTkAgg(fig, master=self.eval_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                # Leave drawing the graph to the teammate
                placeholder_text = f"{selected_model} Confusion Matrix Computed!\n\nRaw Output:\n{cm}\n\n# TODO: Teammate to add graph here."
                tk.Label(self.eval_canvas_frame, text=placeholder_text, bg="white", font=("Consolas", 12), fg="#333333", justify="center").pack(expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate confusion matrix:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictApp(root)
    root.mainloop()