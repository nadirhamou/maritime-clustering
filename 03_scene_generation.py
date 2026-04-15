import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import pickle
    from scipy.spatial.distance import cdist
    import warnings
    warnings.filterwarnings('ignore')

    mo.md("""
    # Maritime Scene Generation

    This notebook generates maritime scenes for SSL training:
    1. Define scene parameters (time window, spatial radius)
    2. Sample vessels as scene centers
    3. Extract nearby vessels within time/space window
    4. Aggregate ocean and weather data for each scene
    5. Create feature vectors for each scene
    """)
    return mo, np, pd, pickle, timedelta


@app.cell
def _(mo):
    mo.md("""
    ## 1. Configuration
    """)
    return


@app.cell
def _():
    # Scene parameters
    SCENE_CONFIG = {
        'time_window_minutes': 20,  # 15-30 minutes as per requirements
        'spatial_radius_nm': 7.5,   # 5-10 nautical miles
        'min_vessels_in_scene': 2,  # Minimum vessels for a valid scene
        'sample_interval_minutes': 30,  # Sample scenes every 30 minutes
        'nm_to_degrees': 1/60,  # Approximate conversion (1 nm ≈ 1/60 degree)
    }

    # Convert spatial radius to degrees
    SCENE_CONFIG['spatial_radius_deg'] = SCENE_CONFIG['spatial_radius_nm'] * SCENE_CONFIG['nm_to_degrees']

    print("Scene Configuration:")
    for key, value in SCENE_CONFIG.items():
        print(f"  {key}: {value}")
    return (SCENE_CONFIG,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Load Preprocessed Data
    """)
    return


@app.cell
def _(pd):
    print("Loading preprocessed data...")
    ais_data = pd.read_pickle('preprocessed_ais.pkl')
    ocean_data = pd.read_pickle('preprocessed_ocean.pkl')
    weather_data = pd.read_pickle('preprocessed_weather.pkl')
    weather_stations = pd.read_pickle('weather_stations.pkl')

    print(f"✓ AIS data: {len(ais_data):,} records")
    print(f"✓ Ocean data: {len(ocean_data):,} records")
    print(f"✓ Weather data: {len(weather_data):,} records")
    return ais_data, ocean_data, weather_data, weather_stations


@app.cell
def _(pickle):
    # Load spatial indices
    with open('spatial_indices.pkl', 'rb') as f:
        spatial_indices = pickle.load(f)

    ais_tree = spatial_indices['ais_tree']
    ocean_tree = spatial_indices['ocean_tree']
    station_tree = spatial_indices['station_tree']

    print("✓ Spatial indices loaded")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Sample Scene Centers
    """)
    return


@app.cell
def _(SCENE_CONFIG, ais_data, pd, timedelta):
    # Sample vessels at regular time intervals
    print("Sampling scene centers...")

    # Get time range
    start_time = ais_data['datetime'].min()
    end_time = ais_data['datetime'].max()

    # Create time bins
    time_bins = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{SCENE_CONFIG['sample_interval_minutes']}min"
    )

    print(f"Time range: {start_time} to {end_time}")
    print(f"Number of time bins: {len(time_bins)}")

    # Sample vessels from each time bin
    scene_centers = []

    for time_bin in time_bins[:1000]:  # Limit to first 1000 for initial testing
        # Get vessels active in this time window
        time_mask = (
            (ais_data['datetime'] >= time_bin - timedelta(minutes=5)) &
            (ais_data['datetime'] <= time_bin + timedelta(minutes=5))
        )

        vessels_in_window = ais_data[time_mask]

        if len(vessels_in_window) > 0:
            # Sample one vessel as scene center
            center = vessels_in_window.sample(n=1).iloc[0]
            scene_centers.append({
                'scene_id': len(scene_centers),
                'center_mmsi': center['sourcemmsi'],
                'center_time': time_bin,
                'center_lon': center['lon'],
                'center_lat': center['lat'],
                'center_speed': center['speedoverground'],
                'center_course': center['courseoverground']
            })

    scene_centers_df = pd.DataFrame(scene_centers)
    print(f"Generated {len(scene_centers_df)} scene centers")
    return (scene_centers_df,)


@app.cell
def _(scene_centers_df):
    scene_centers_df.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Extract Vessels for Each Scene
    """)
    return


@app.cell
def _(SCENE_CONFIG, ais_data, np, timedelta):
    def extract_scene_vessels(scene_center):
        """Extract all vessels within spatial and temporal window of scene center"""

        # Temporal window
        time_start = scene_center['center_time'] - timedelta(minutes=SCENE_CONFIG['time_window_minutes']/2)
        time_end = scene_center['center_time'] + timedelta(minutes=SCENE_CONFIG['time_window_minutes']/2)

        # Spatial window (approximate using lat/lon box)
        lon_min = scene_center['center_lon'] - SCENE_CONFIG['spatial_radius_deg']
        lon_max = scene_center['center_lon'] + SCENE_CONFIG['spatial_radius_deg']
        lat_min = scene_center['center_lat'] - SCENE_CONFIG['spatial_radius_deg']
        lat_max = scene_center['center_lat'] + SCENE_CONFIG['spatial_radius_deg']

        # Filter AIS data
        mask = (
            (ais_data['datetime'] >= time_start) &
            (ais_data['datetime'] <= time_end) &
            (ais_data['lon'] >= lon_min) &
            (ais_data['lon'] <= lon_max) &
            (ais_data['lat'] >= lat_min) &
            (ais_data['lat'] <= lat_max)
        )

        scene_vessels = ais_data[mask].copy()

        # Calculate relative positions and times
        scene_vessels['rel_lon'] = scene_vessels['lon'] - scene_center['center_lon']
        scene_vessels['rel_lat'] = scene_vessels['lat'] - scene_center['center_lat']
        scene_vessels['rel_time'] = (scene_vessels['datetime'] - scene_center['center_time']).dt.total_seconds()

        # Calculate distance from center
        scene_vessels['distance_from_center'] = np.sqrt(
            scene_vessels['rel_lon']**2 + scene_vessels['rel_lat']**2
        )

        return scene_vessels

    print("Extracting vessels for each scene...")
    return (extract_scene_vessels,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Extract Ocean Data for Each Scene
    """)
    return


@app.cell
def _(SCENE_CONFIG, np, ocean_data, timedelta):
    def extract_scene_ocean(scene_center):
        """Extract ocean data for scene"""

        # Temporal window (use closest time)
        time_start = scene_center['center_time'] - timedelta(hours=3)
        time_end = scene_center['center_time'] + timedelta(hours=3)

        # Spatial window
        lon_min = scene_center['center_lon'] - SCENE_CONFIG['spatial_radius_deg']
        lon_max = scene_center['center_lon'] + SCENE_CONFIG['spatial_radius_deg']
        lat_min = scene_center['center_lat'] - SCENE_CONFIG['spatial_radius_deg']
        lat_max = scene_center['center_lat'] + SCENE_CONFIG['spatial_radius_deg']

        # Filter ocean data
        mask = (
            (ocean_data['datetime'] >= time_start) &
            (ocean_data['datetime'] <= time_end) &
            (ocean_data['lon'] >= lon_min) &
            (ocean_data['lon'] <= lon_max) &
            (ocean_data['lat'] >= lat_min) &
            (ocean_data['lat'] <= lat_max)
        )

        scene_ocean = ocean_data[mask]

        if len(scene_ocean) > 0:
            # Aggregate ocean features
            ocean_features = {
                'ocean_hs_mean': scene_ocean['hs'].mean(),
                'ocean_hs_std': scene_ocean['hs'].std(),
                'ocean_lm_mean': scene_ocean['lm'].mean(),
                'ocean_dir_mean': scene_ocean['dir'].mean(),
                'ocean_wlv_mean': scene_ocean['wlv'].mean(),
            }
        else:
            ocean_features = {
                'ocean_hs_mean': np.nan,
                'ocean_hs_std': np.nan,
                'ocean_lm_mean': np.nan,
                'ocean_dir_mean': np.nan,
                'ocean_wlv_mean': np.nan,
            }

        return ocean_features

    print("Ocean extraction function defined")
    return (extract_scene_ocean,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Extract Weather Data for Each Scene
    """)
    return


@app.cell
def _(np, timedelta, weather_data, weather_stations):
    def extract_scene_weather(scene_center):
        """Extract weather data for scene from nearest station"""

        # Find nearest weather station
        distances = np.sqrt(
            (weather_stations['longitude'] - scene_center['center_lon'])**2 +
            (weather_stations['latitude'] - scene_center['center_lat'])**2
        )
        nearest_station_id = weather_stations.iloc[distances.argmin()]['id_station']

        # Get weather data from nearest station
        time_start = scene_center['center_time'] - timedelta(hours=1)
        time_end = scene_center['center_time'] + timedelta(hours=1)

        mask = (
            (weather_data['id_station'] == nearest_station_id) &
            (weather_data['datetime'] >= time_start) &
            (weather_data['datetime'] <= time_end)
        )

        scene_weather = weather_data[mask]

        if len(scene_weather) > 0:
            # Get closest observation
            closest_obs = scene_weather.iloc[(scene_weather['datetime'] - scene_center['center_time']).abs().argmin()]

            weather_features = {
                'weather_temp': closest_obs['T'],
                'weather_pressure': closest_obs['P'],
                'weather_humidity': closest_obs['U'],
                'weather_wind_speed': closest_obs['Ff'],
                'weather_visibility': closest_obs['VV'],
                'weather_station_distance': distances.min()
            }
        else:
            weather_features = {
                'weather_temp': np.nan,
                'weather_pressure': np.nan,
                'weather_humidity': np.nan,
                'weather_wind_speed': np.nan,
                'weather_visibility': np.nan,
                'weather_station_distance': distances.min()
            }

        return weather_features

    print("Weather extraction function defined")
    return (extract_scene_weather,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Generate Complete Scenes
    """)
    return


@app.cell
def _(
    SCENE_CONFIG,
    extract_scene_ocean,
    extract_scene_vessels,
    extract_scene_weather,
    np,
    pd,
    scene_centers_df,
):
    def generate_scene_features(scene_center):
        """Generate complete feature vector for a scene"""

        # Extract vessels
        vessels = extract_scene_vessels(scene_center)

        # Check minimum vessel requirement
        if len(vessels) < SCENE_CONFIG['min_vessels_in_scene']:
            return None

        # Vessel-level features (aggregated)
        vessel_features = {
            'num_vessels': len(vessels),
            'num_unique_vessels': vessels['sourcemmsi'].nunique(),
            'speed_mean': vessels['speedoverground'].mean(),
            'speed_std': vessels['speedoverground'].std(),
            'speed_max': vessels['speedoverground'].max(),
            'speed_min': vessels['speedoverground'].min(),
            'course_std': vessels['courseoverground'].std(),
            'distance_mean': vessels['distance_from_center'].mean(),
            'distance_max': vessels['distance_from_center'].max(),
            'length_mean': vessels['length'].mean(),
            'beam_mean': vessels['beam'].mean(),
        }

        # Calculate vessel density
        vessel_features['vessel_density'] = len(vessels) / (np.pi * SCENE_CONFIG['spatial_radius_deg']**2)

        # Extract ocean features
        ocean_features = extract_scene_ocean(scene_center)

        # Extract weather features
        weather_features = extract_scene_weather(scene_center)

        # Combine all features
        scene_features = {
            'scene_id': scene_center['scene_id'],
            'center_mmsi': scene_center['center_mmsi'],
            'center_time': scene_center['center_time'],
            'center_lon': scene_center['center_lon'],
            'center_lat': scene_center['center_lat'],
            **vessel_features,
            **ocean_features,
            **weather_features
        }

        return scene_features

    print("Generating scenes...")
    scenes = []

    for idx, scene_center in scene_centers_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing scene {idx}/{len(scene_centers_df)}...")

        scene_features = generate_scene_features(scene_center)

        if scene_features is not None:
            scenes.append(scene_features)

    scenes_df = pd.DataFrame(scenes)
    print(f"\n✓ Generated {len(scenes_df)} valid scenes")
    return (scenes_df,)


@app.cell
def _(scenes_df):
    scenes_df.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Scene Statistics
    """)
    return


@app.cell
def _(mo, scenes_df):
    mo.md(f"""
    ### Scene Generation Summary

    - **Total Scenes**: {len(scenes_df):,}
    - **Vessels per Scene**: {scenes_df['num_vessels'].mean():.1f} ± {scenes_df['num_vessels'].std():.1f}
    - **Unique Vessels per Scene**: {scenes_df['num_unique_vessels'].mean():.1f}
    - **Average Speed**: {scenes_df['speed_mean'].mean():.2f} knots
    - **Vessel Density**: {scenes_df['vessel_density'].mean():.4f} vessels/deg²
    """)
    return


@app.cell
def _(scenes_df):
    # Feature statistics
    feature_cols = [col for col in scenes_df.columns if col not in ['scene_id', 'center_mmsi', 'center_time', 'center_lon', 'center_lat']]
    scenes_df[feature_cols].describe()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Visualize Scene Distribution
    """)
    return


@app.cell
def _(scenes_df):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Number of vessels per scene
    axes[0, 0].hist(scenes_df['num_vessels'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Number of Vessels')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Vessels per Scene Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Average speed per scene
    axes[0, 1].hist(scenes_df['speed_mean'].dropna(), bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Average Speed (knots)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Scene Average Speed Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Vessel density
    axes[1, 0].hist(scenes_df['vessel_density'], bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Vessel Density (vessels/deg²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Vessel Density Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Scene locations
    axes[1, 1].scatter(scenes_df['center_lon'], scenes_df['center_lat'], alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].set_title('Scene Center Locations')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Save Generated Scenes
    """)
    return


@app.cell
def _(scenes_df):
    # Save scenes
    print("Saving generated scenes...")
    scenes_df.to_pickle('maritime_scenes.pkl')
    scenes_df.to_csv('maritime_scenes.csv', index=False)
    print("✓ Saved maritime_scenes.pkl and maritime_scenes.csv")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Summary

    ### Scene Generation Complete

    **Output Files:**
    - `maritime_scenes.pkl`: Complete scene features for SSL training
    - `maritime_scenes.csv`: Human-readable scene data

    **Scene Characteristics:**
    - Each scene represents a 20-minute window around a vessel
    - Spatial radius of 7.5 nautical miles
    - Includes vessel dynamics, ocean conditions, and weather data
    - Aggregated features for efficient model training

    **Next Steps:**
    - Build SSL model (autoencoder or contrastive learning)
    - Train model to generate scene embeddings
    - Apply clustering (HDBSCAN) to identify maritime activity patterns
    """)
    return


if __name__ == "__main__":
    app.run()
