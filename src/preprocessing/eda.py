import pandas as pd

RAW_PATH = "data/raw/"

constructor_standings = pd.read_csv(RAW_PATH + "constructor_standings.csv")
constructors = pd.read_csv(RAW_PATH + "constructors.csv")
driver_standings = pd.read_csv(RAW_PATH + "driver_standings.csv")
races = pd.read_csv(RAW_PATH + "races.csv")
drivers = pd.read_csv(RAW_PATH + "drivers.csv")
qualifying = pd.read_csv(RAW_PATH + "qualifying.csv")
results = pd.read_csv(RAW_PATH + "results.csv")
status = pd.read_csv(RAW_PATH + "status.csv")

# frames
frames = {
    "constructor_standings" : constructor_standings,
    "constructors" : constructors,
    "driver_standings" : driver_standings,
    "races" : races,
    "drivers" : drivers,
    "qualifying" : qualifying,
    "results" : results,
    "status" : status
}

# for name, f in frames.items():
#     print(f"\n{name}")
#     print("shape:", f.shape)
#     print("columns:", list(f.columns))
#     print(f.head(8))

# for name, df in frames.items():
#     print(f"\n--- {name} ---")
#     print(df.isnull().sum())  # counts actual NaN per column

#     # This dataset often uses the STRING '\N' instead of a real NaN
#     for col in df.columns:
#         if df[col].dtype == object:  # only check text-type columns
#             count_backslash_n = (df[col] == r'\N').sum()
#             if count_backslash_n > 0:
#                 print(f"  '{col}' has {count_backslash_n} '\\N' placeholder values")

# print(races["year"].min(), "to", races["year"].max())
# print("Total races:", races.shape[0])
# print("Races per year (last 5 years):")
# print(races["year"].value_counts().sort_index().tail(5))

# print("""
# JOIN MAP:
# results.raceId          -> races.raceId
# results.driverId        -> drivers.driverId
# results.constructorId   -> constructors.constructorId
# results.statusId        -> status.statusId
# qualifying.raceId + driverId       -> results.raceId + driverId
# driver_standings.raceId + driverId -> results.raceId + driverId
# constructor_standings.raceId + constructorId -> results.raceId + constructorId
# """)

# print(results[["position", "positionOrder", "statusId"]].head(10))
# print(status.head(20))