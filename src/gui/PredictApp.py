import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'prediction'))
from predict import predict_all, FEATURES

df = pd.read_csv("data/processed/final.csv")

class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("F1 Podium Predictor")
        self.root.geometry("500x600")

        self.build_widgets()

    def build_widgets(self):
        tk.Label(self.root, text="Season: ").pack(pady=(20, 0))
        self.season_var = tk.StringVar()
        seasons = sorted(df['year'].unique().tolist(), reverse=True)
        self.season_dropdown = ttk.Combobox(self.root, textvariable=self.season_var, values=seasons, state="readonly")
        self.season_dropdown.pack()
        self.season_dropdown.bind("<<ComboboxSelected>>", self.on_season_selected)

        tk.Label(self.root, text="Grand Prix: ").pack(pady=(20, 0))
        self.race_var = tk.StringVar()
        self.race_dropdown = ttk.Combobox(self.root, textvariable=self.race_var, values=[], state="readonly")
        self.race_dropdown.pack()
        self.race_dropdown.bind("<<ComboboxSelected>>", self.on_race_selected)

        tk.Label(self.root, text="Driver: ").pack(pady=(20, 0))
        self.driver_var = tk.StringVar()
        self.driver_dropdown = ttk.Combobox(self.root, textvariable=self.driver_var, values=[], state="readonly")
        self.driver_dropdown.pack()

        tk.Label(self.root, text="Algorithm:").pack(pady=(20, 0))
        self.algo_var = tk.StringVar()
        self.algo_dropdown = ttk.Combobox(self.root, textvariable=self.algo_var, 
                                        values=["Random Forest", "SVM", "KNN", "Compare All"], state="readonly"
                                        )
        self.algo_dropdown.current(3)
        self.algo_dropdown.pack()

        tk.Button(self.root, text="Predict", command=self.on_predict, bg="#e10600", fg="white", 
                  font=("Arial", 12, "bold")).pack(pady=30)

        self.result_label = tk.Label(self.root, text="", justify="left", font=("Consolas", 11))
        self.result_label.pack(pady=10)

    def on_season_selected(self, event=None):
        selected_year = int(self.season_var.get())
        races_in_season = df[df['year'] == selected_year]['name'].unique().tolist()
        self.race_dropdown['values'] = races_in_season
        self.race_var.set("")
        self.driver_dropdown['values'] = []
        self.driver_var.set("")

    def on_race_selected(self, event=None):
        selected_year = int(self.season_var.get())
        selected_race = self.race_var.get()
        subset = df[(df['year'] == selected_year) & (df['name'] == selected_race)]
        drivers = (subset['forename'] + " " + subset['surname']).unique().tolist()
        self.driver_dropdown['values'] = drivers
        self.driver_var.set("")

    def on_predict(self):
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

        output_lines = [f"Driver: {selected_driver}", f"Race: {selected_year} {selected_race} {selected_driver}"]
        for model_name, result in results.items():
            conf = result['probability']
            conf_str = f"{conf}%" if conf is not None else "N/A"
            output_lines.append(f"{model_name}: {result['prediction']} ({conf_str} confidence)")

        self.result_label.config(text="\n".join(output_lines))

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictApp(root)
    root.mainloop()