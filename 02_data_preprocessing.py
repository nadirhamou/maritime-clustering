import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from scipy.spatial import cKDTree
    import pickle
    import warnings
    warnings.filterwarnings('ignore')

    mo.md("""
    # Maritime Scene Clustering - Data Preprocessing

    This notebook preprocesses the raw maritime data for scene generation:
    1. Load and clean AIS, Ocean, and Weather data
    2. Handle missing values and outliers
    3. Create spatial and temporal indices
    4. Prepare data structures for efficient scene extraction
    """)
    return cKDTree, mo, np, pd, pickle


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Raw Data
    """)
    return


@app.cell
def _(pd):
    # Load AIS data
    print("Loading AIS data...")
    ais_dynamic = pd.read_csv('AIS_DATA/nari_dynamic.csv')
    ais_static = pd.read_csv('AIS_DATA/nari_static.csv')

    print(f"AIS Dynamic: {len(ais_dynamic):,} records")
    print(f"AIS Static: {len(ais_static):,} records")
    return


@app.cell
def _(pd):
    # Load Ocean data
    print("Loading Ocean data...")
    ocean_files = [
        'OCEAN_DATA/oc_october.csv',
        'OCEAN_DATA/oc_november.csv',
        'OCEAN_DATA/oc_december.csv',
        'OCEAN_DATA/oc_january.csv',
        'OCEAN_DATA/oc_february.csv',
        'OCEAN_DATA/oc_march.csv'
    ]
    ocean_data = pd.concat([pd.read_csv(f) for f in ocean_files], ignore_index=True)
    print(f"Ocean Data: {len(ocean_data):,} records")
    return (ocean_data,)


@app.cell
def _(pd):
    # Load Weather data
    print("Loading Weather data...")
    weather_obs = pd.read_csv('WEATHER_DATA/table_wheatherObservation.csv')
    weather_stations = pd.read_csv('WEATHER_DATA/table_weatherStation.csv')

    print(f"Weather Observations: {len(weather_obs):,} records")
    print(f"Weather Stations: {len(weather_stations)} stations")
    return weather_obs, weather_stations


@app.cell
def _(mo):
    mo.md("""
    ## 2. Clean and Preprocess AIS Data
    """)
    return


@app.cell
def _(np, pd):
    # Load & clean AIS data
    print("Loading AIS data...")
    ais_dynamic_raw = pd.read_csv('AIS_DATA/nari_dynamic.csv')
    ais_static_raw = pd.read_csv('AIS_DATA/nari_static.csv')

    # Convert timestamps
    ais_dynamic_raw['datetime'] = pd.to_datetime(ais_dynamic_raw['t'], unit='s')

    # Handle missing/invalid values
    ais_dynamic_raw.loc[ais_dynamic_raw['speedoverground'] > 102.2, 'speedoverground'] = np.nan
    ais_dynamic_raw.loc[ais_dynamic_raw['speedoverground'] < 0, 'speedoverground'] = np.nan
    ais_dynamic_raw.loc[ais_dynamic_raw['courseoverground'] >= 360, 'courseoverground'] = np.nan
    ais_dynamic_raw.loc[ais_dynamic_raw['courseoverground'] < 0, 'courseoverground'] = np.nan
    ais_dynamic_raw.loc[ais_dynamic_raw['trueheading'] == 511, 'trueheading'] = np.nan
    ais_dynamic_raw.loc[ais_dynamic_raw['trueheading'] >= 360, 'trueheading'] = np.nan
    ais_dynamic_raw.loc[ais_dynamic_raw['rateofturn'] == -127, 'rateofturn'] = np.nan

    # Filter to geographic area of interest
    ais_dynamic_clean = ais_dynamic_raw[
        (ais_dynamic_raw['lon'] >= -10) & (ais_dynamic_raw['lon'] <= 0) &
        (ais_dynamic_raw['lat'] >= 45) & (ais_dynamic_raw['lat'] <= 51)
    ].copy()

    print(f"AIS Dynamic (raw): {len(ais_dynamic_raw):,} records")
    print(f"AIS Static: {len(ais_static_raw):,} records")
    print(f"Cleaned AIS data: {len(ais_dynamic_clean):,} records")
    return ais_dynamic_clean, ais_static_raw


@app.cell
def _(ais_dynamic_clean, ais_static_raw):
    # Sample for development
    ais_sample = ais_dynamic_clean.sample(n=10_000, random_state=42)

    # Ensure same type for join key
    ais_sample['sourcemmsi'] = ais_sample['sourcemmsi'].astype(str)
    ais_static_raw['sourcemmsi'] = ais_static_raw['sourcemmsi'].astype(str)

    # Merge static info
    ais_merged = ais_sample.merge(
        ais_static_raw[['sourcemmsi', 'shiptype', 'tobow', 'tostern', 'toport', 'tostarboard']],
        on='sourcemmsi',
        how='left'
    )

    # Calculate vessel length and beam (noms de colonnes corrects)
    ais_merged['length'] = ais_merged['tobow'] + ais_merged['tostern']
    ais_merged['beam'] = ais_merged['toport'] + ais_merged['tostarboard']

    print(f"Sample size: {len(ais_sample):,} records")
    print(f"Merged AIS data: {len(ais_merged):,} records")
    return (ais_merged,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Clean and Preprocess Ocean Data
    """)
    return


@app.cell
def _(np, ocean_data, pd):
    # Convert timestamps
    ocean_data['datetime'] = pd.to_datetime(ocean_data['ts'], unit='s')

    # Handle undefined values
    ocean_data.loc[ocean_data['dpt'] == -16384, 'dpt'] = np.nan
    ocean_data.loc[ocean_data['wlv'] == -327.67, 'wlv'] = np.nan
    ocean_data.loc[ocean_data['hs'] == -65.534, 'hs'] = np.nan
    ocean_data.loc[ocean_data['lm'] == -32767, 'lm'] = np.nan
    ocean_data.loc[ocean_data['dir'] == -3276.7, 'dir'] = np.nan

    # Filter to geographic area + sample
    ocean_clean = ocean_data[
        (ocean_data['lon'] >= -10) & (ocean_data['lon'] <= 0) &
        (ocean_data['lat'] >= 45) & (ocean_data['lat'] <= 51)
    ].sample(n=1000, random_state=42).copy()

    print(f"Cleaned Ocean data: {len(ocean_clean):,} records")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Clean and Preprocess Weather Data
    """)
    return


@app.cell
def _(np, pd, weather_obs):
    # Convert timestamps
    weather_obs['datetime'] = pd.to_datetime(weather_obs['local_time'], unit='s')

    # Handle undefined values (-65536 indicates missing data)
    weather_cols = ['T', 'Tn', 'Tx', 'P', 'U', 'Ff', 'ff10', 'ff3', 'VV', 'Td', 'RRR', 'tR']
    for col in weather_cols:
        if col in weather_obs.columns:
            weather_obs.loc[weather_obs[col] == -65536, col] = np.nan

    print(f"Cleaned Weather data: {len(weather_obs):,} records")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Create Spatial Indices for Efficient Queries
    """)
    return


@app.cell
def _(ais_merged, cKDTree):
    # Create spatial index for AIS data
    print("Creating spatial index for AIS data...")
    ais_coords = ais_merged[['lon', 'lat']].values
    ais_tree = cKDTree(ais_coords)

    print(f"AIS spatial index created with {len(ais_coords):,} points")
    return (ais_tree,)


@app.cell
def _(cKDTree, ocean_data):
    # Create spatial index for Ocean data
    print("Creating spatial index for Ocean data...")
    ocean_coords = ocean_data[['lon', 'lat']].values
    ocean_tree = cKDTree(ocean_coords)

    print(f"Ocean spatial index created with {len(ocean_coords):,} points")
    return (ocean_tree,)


@app.cell
def _(cKDTree, weather_stations):
    # Create spatial index for Weather stations
    print("Creating spatial index for Weather stations...")
    station_coords = weather_stations[['longitude', 'latitude']].values
    station_tree = cKDTree(station_coords)

    print(f"Weather station spatial index created with {len(station_coords)} points")
    return (station_tree,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Create Temporal Indices
    """)
    return


@app.cell
def _(ais_merged):
    # Sort AIS data by time for efficient temporal queries
    ais_merged_sorted = ais_merged.sort_values('datetime').reset_index(drop=True)

    # Create time-based index
    ais_merged_sorted['time_idx'] = range(len(ais_merged_sorted))

    print(f"AIS data sorted by time: {len(ais_merged_sorted):,} records")
    return (ais_merged_sorted,)


@app.cell
def _(ocean_data):
    # Sort Ocean data by time
    ocean_sorted = ocean_data.sort_values('datetime').reset_index(drop=True)
    ocean_sorted['time_idx'] = range(len(ocean_sorted))

    print(f"Ocean data sorted by time: {len(ocean_sorted):,} records")
    return (ocean_sorted,)


@app.cell
def _(weather_obs):
    # Sort Weather data by time
    weather_sorted = weather_obs.sort_values('datetime').reset_index(drop=True)
    weather_sorted['time_idx'] = range(len(weather_sorted))

    print(f"Weather data sorted by time: {len(weather_sorted):,} records")
    return (weather_sorted,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Feature Engineering
    """)
    return


@app.cell
def _(ais_merged_sorted, np):

    # Calculate velocity components
    ais_merged_sorted['vx'] = ais_merged_sorted['speedoverground'] * np.cos(np.radians(ais_merged_sorted['courseoverground']))
    ais_merged_sorted['vy'] = ais_merged_sorted['speedoverground'] * np.sin(np.radians(ais_merged_sorted['courseoverground']))

    # Calculate acceleration (change in velocity over time)
    ais_merged_sorted['time_diff'] = ais_merged_sorted.groupby('sourcemmsi')['datetime'].diff().dt.total_seconds()
    ais_merged_sorted['speed_diff'] = ais_merged_sorted.groupby('sourcemmsi')['speedoverground'].diff()
    ais_merged_sorted['acceleration'] = ais_merged_sorted['speed_diff'] / ais_merged_sorted['time_diff']

    # Calculate distance traveled
    ais_merged_sorted['lon_diff'] = ais_merged_sorted.groupby('sourcemmsi')['lon'].diff()
    ais_merged_sorted['lat_diff'] = ais_merged_sorted.groupby('sourcemmsi')['lat'].diff()

    print("Feature engineering complete")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Save Preprocessed Data
    """)
    return


@app.cell
def _(ais_merged_sorted):
    # Save preprocessed AIS data
    print("Saving preprocessed AIS data...")
    ais_merged_sorted.to_pickle('preprocessed_ais.pkl')
    print("✓ Saved preprocessed_ais.pkl")
    return


@app.cell
def _(ocean_sorted):
    # Save preprocessed Ocean data
    print("Saving preprocessed Ocean data...")
    ocean_sorted.to_pickle('preprocessed_ocean.pkl')
    print("✓ Saved preprocessed_ocean.pkl")
    return


@app.cell
def _(weather_sorted):
    # Save preprocessed Weather data
    print("Saving preprocessed Weather data...")
    weather_sorted.to_pickle('preprocessed_weather.pkl')
    print("✓ Saved preprocessed_weather.pkl")
    return


@app.cell
def _(ais_tree, ocean_tree, pickle, station_tree):
    # Save spatial indices
    print("Saving spatial indices...")
    with open('spatial_indices.pkl', 'wb') as f:
        pickle.dump({
            'ais_tree': ais_tree,
            'ocean_tree': ocean_tree,
            'station_tree': station_tree
        }, f)
    print("✓ Saved spatial_indices.pkl")
    return


@app.cell
def _(weather_stations):
    # Save weather stations info
    weather_stations.to_pickle('weather_stations.pkl')
    print("✓ Saved weather_stations.pkl")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Summary

    ### Preprocessing Complete

    **Files Created:**
    - `preprocessed_ais.pkl`: Cleaned and feature-engineered AIS data
    - `preprocessed_ocean.pkl`: Cleaned ocean state data
    - `preprocessed_weather.pkl`: Cleaned weather observation data
    - `spatial_indices.pkl`: KD-trees for efficient spatial queries
    - `weather_stations.pkl`: Weather station locations

    **Data Quality:**
    - Removed invalid positions and outlier values
    - Handled missing data indicators
    - Created velocity and acceleration features
    - Built spatial and temporal indices for efficient scene extraction

    **Next Steps:**
    - Generate maritime scenes (15-30 min windows around vessels)
    - Extract features for each scene
    - Build SSL model for scene encoding
    """)
    return


if __name__ == "__main__":
    app.run()
