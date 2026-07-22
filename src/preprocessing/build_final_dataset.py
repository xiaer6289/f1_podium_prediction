import pandas as pd

RAW_PATH = "data/raw/"

constructor_standings = pd.read_csv(RAW_PATH + "constructor_standings.csv", na_values=r'\N')
constructors = pd.read_csv(RAW_PATH + "constructors.csv", na_values=r'\N')
driver_standings = pd.read_csv(RAW_PATH + "driver_standings.csv", na_values=r'\N')
races = pd.read_csv(RAW_PATH + "races.csv", na_values=r'\N')
drivers = pd.read_csv(RAW_PATH + "drivers.csv", na_values=r'\N')
qualifying = pd.read_csv(RAW_PATH + "qualifying.csv", na_values=r'\N')
results = pd.read_csv(RAW_PATH + "results.csv", na_values=r'\N')
status = pd.read_csv(RAW_PATH + "status.csv", na_values=r'\N')

ds = driver_standings.merge(races[['raceId', 'year', 'round']], on='raceId')
ds = ds.sort_values(['driverId', 'year', 'round'])
ds[['driver_points_before', 'driver_position_before', 'driver_wins_before']] = (
    ds.groupby('driverId')[['points', 'position', 'wins']].shift(1)
)
driver_standings_shifted = ds[['raceId', 'driverId', 'driver_points_before', 'driver_position_before', 'driver_wins_before']]


cs = constructor_standings.merge(races[['raceId', 'year', 'round']], on='raceId')
cs = cs.sort_values(['constructorId', 'year', 'round'])
cs[['constructor_points_before', 'constructor_position_before', 'constructor_wins_before']] = (
    cs.groupby('constructorId')[['points', 'position', 'wins']].shift(1)
)
constructor_standings_shifted = cs[['raceId', 'constructorId', 'constructor_points_before', 'constructor_position_before', 'constructor_wins_before']]

# filter
final = results.merge(races[['raceId', 'year', 'round', 'name']], on='raceId')
final = final[final['year'] >= 2010]

final = final.merge(
    qualifying[['raceId', 'driverId', 'position']]
    .rename(columns={
        'position': 'qualifying_position'}),
        on=['raceId', 'driverId'], how='left'
)

final = final.merge(drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')

final = final.merge(
    constructors[['constructorId', 'name']]
    .rename(columns={
        'name': 'constructor_name'}),
        on=['constructorId'], how = 'left'
)

final = final.merge(driver_standings_shifted, on=['raceId', 'driverId'], how='left')
final = final.merge(constructor_standings_shifted, on=['raceId', 'constructorId'], how='left')

final['driver_points_before'] = final['driver_points_before'].fillna(0)
final['driver_wins_before'] = final['driver_wins_before'].fillna(0)
final['driver_position_before'] = final['driver_position_before'].fillna(final['driver_position_before'].max() + 1)

final['constructor_points_before'] = final['constructor_points_before'].fillna(0)
final['constructor_wins_before'] = final['constructor_wins_before'].fillna(0)
final['constructor_position_before'] = final['constructor_position_before'].fillna(final['constructor_position_before'].max() + 1)

final['qualifying_position'] = final['qualifying_position'].fillna(final['qualifying_position'].max() + 1)

# build target column
final['podium'] = (final['positionOrder'] <= 3).astype(int)

# save relevant column only
final = final[[
    'raceId', 'year', 'round', 'name', 
    'driverId', 'forename', 'surname', 
    'constructorId', 'constructor_name',
    'grid', 'qualifying_position', 
    'driver_points_before', 'driver_position_before', 'driver_wins_before',
    'constructor_points_before', 'constructor_position_before', 'constructor_wins_before',
    'podium'
]]

final.to_csv("data/processed/final.csv", index=False)
print("Saved final.csv", final.shape)
print(final['podium'].value_counts())