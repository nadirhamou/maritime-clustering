import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')

    mo.md("""
    # Maritime Scene Clustering - Data Exploration

    This notebook explores the maritime datasets including:
    - **AIS Data**: Vessel tracking information (position, speed, course, etc.)
    - **Ocean Data**: Sea state forecasts (wave height, direction, etc.)
    - **Weather Data**: Coastal weather observations

    ## Project Goal
    Build an SSL-based AI model to cluster maritime scenes and identify operational activities
    such as bunkering, port operations, smuggling, etc.
    """)
    return mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load AIS Data
    """)
    return


@app.cell
def _(pd):
    # Load AIS dynamic data
    ais_dynamic = pd.read_csv('AIS_DATA/nari_dynamic.csv')
    ais_static = pd.read_csv('AIS_DATA/nari_static.csv')

    print(f"AIS Dynamic shape: {ais_dynamic.shape}")
    print(f"AIS Static shape: {ais_static.shape}")
    return ais_dynamic, ais_static


@app.cell
def _(ais_dynamic, mo, pd):
    # Convert timestamp to datetime
    ais_dynamic['datetime'] = pd.to_datetime(ais_dynamic['t'], unit='s')

    mo.md(f"""
    ### AIS Dynamic Data Overview
    - **Records**: {len(ais_dynamic):,}
    - **Unique Vessels (MMSI)**: {ais_dynamic['sourcemmsi'].nunique():,}
    - **Time Range**: {ais_dynamic['datetime'].min()} to {ais_dynamic['datetime'].max()}
    - **Duration**: {(ais_dynamic['datetime'].max() - ais_dynamic['datetime'].min()).days} days
    """)
    return


@app.cell
def _(ais_dynamic):
    ais_dynamic.head(10)
    return


@app.cell
def _(ais_dynamic, mo):
    # Basic statistics
    stats = ais_dynamic[['speedoverground', 'courseoverground', 'trueheading', 'lon', 'lat']].describe()

    mo.md("### AIS Data Statistics")
    return (stats,)


@app.cell
def _(stats):
    stats
    return


@app.cell
def _(ais_static, mo):
    mo.md(f"""
    ### AIS Static Data Overview
    - **Records**: {len(ais_static):,}
    - **Unique Vessels**: {ais_static['sourcemmsi'].nunique():,}
    """)
    return


@app.cell
def _(ais_static):
    ais_static.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Load Ocean Data
    """)
    return


@app.cell
def _(pd):
    # Load ocean data from all months
    ocean_files = [
        'OCEAN_DATA/oc_october.csv',
        'OCEAN_DATA/oc_november.csv', 
        'OCEAN_DATA/oc_december.csv',
        'OCEAN_DATA/oc_january.csv',
        'OCEAN_DATA/oc_february.csv',
        'OCEAN_DATA/oc_march.csv'
    ]

    ocean_data = pd.concat([pd.read_csv(f) for f in ocean_files], ignore_index=True)
    print(f"Ocean Data shape: {ocean_data.shape}")
    return (ocean_data,)


@app.cell
def _(mo, ocean_data, pd):
    ocean_data['datetime'] = pd.to_datetime(ocean_data['ts'], unit='s')

    mo.md(f"""
    ### Ocean Data Overview
    - **Records**: {len(ocean_data):,}
    - **Unique Locations**: {ocean_data[['lon', 'lat']].drop_duplicates().shape[0]:,}
    - **Time Range**: {ocean_data['datetime'].min()} to {ocean_data['datetime'].max()}
    - **Parameters**: Wave height (hs), wave length (lm), wave direction (dir), sea level (wlv)
    """)
    return


@app.cell
def _(ocean_data):
    ocean_data.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Load Weather Data
    """)
    return


@app.cell
def _(pd):
    # Load weather data
    weather_obs = pd.read_csv('WEATHER_DATA/table_wheatherObservation.csv')
    weather_stations = pd.read_csv('WEATHER_DATA/table_weatherStation.csv')

    print(f"Weather Observations shape: {weather_obs.shape}")
    print(f"Weather Stations shape: {weather_stations.shape}")
    return weather_obs, weather_stations


@app.cell
def _(mo, pd, weather_obs):
    weather_obs['datetime'] = pd.to_datetime(weather_obs['local_time'], unit='s')

    mo.md(f"""
    ### Weather Data Overview
    - **Observations**: {len(weather_obs):,}
    - **Stations**: {weather_obs['id_station'].nunique()}
    - **Time Range**: {weather_obs['datetime'].min()} to {weather_obs['datetime'].max()}
    - **Parameters**: Temperature, pressure, humidity, wind speed/direction, visibility
    """)
    return


@app.cell
def _(weather_obs):
    weather_obs.head(10)
    return


@app.cell
def _(weather_stations):
    weather_stations
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Spatial Distribution Visualization
    """)
    return


@app.cell
def _(ais_dynamic, plt):
    # Plot vessel positions
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sample data for visualization (too many points otherwise)
    sample = ais_dynamic.sample(min(50000, len(ais_dynamic)))

    scatter = ax.scatter(sample['lon'], sample['lat'], 
                        c=sample['speedoverground'], 
                        cmap='viridis', alpha=0.3, s=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Vessel Positions (colored by speed)')
    plt.colorbar(scatter, label='Speed over Ground (knots)')
    ax.grid(True, alpha=0.3)

    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Temporal Distribution
    """)
    return


@app.cell
def _(ais_dynamic, plt):
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 8))

    # Messages per day
    daily_counts = ais_dynamic.groupby(ais_dynamic['datetime'].dt.date).size()
    ax2a.plot(daily_counts.index, daily_counts.values)
    ax2a.set_xlabel('Date')
    ax2a.set_ylabel('Number of AIS Messages')
    ax2a.set_title('AIS Messages per Day')
    ax2a.grid(True, alpha=0.3)
    ax2a.tick_params(axis='x', rotation=45)

    # Messages per hour of day
    hourly_counts = ais_dynamic.groupby(ais_dynamic['datetime'].dt.hour).size()
    ax2b.bar(hourly_counts.index, hourly_counts.values)
    ax2b.set_xlabel('Hour of Day')
    ax2b.set_ylabel('Number of AIS Messages')
    ax2b.set_title('AIS Messages by Hour of Day')
    ax2b.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Speed and Course Distribution
    """)
    return


@app.cell
def _(ais_dynamic, plt):
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Speed distribution
    ax1.hist(ais_dynamic['speedoverground'].dropna(), bins=50, edgecolor='black')
    ax1.set_xlabel('Speed over Ground (knots)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Speed Distribution')
    ax1.grid(True, alpha=0.3)

    # Course distribution (polar plot would be better but histogram for simplicity)
    ax2.hist(ais_dynamic['courseoverground'].dropna(), bins=36, edgecolor='black')
    ax2.set_xlabel('Course over Ground (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Course Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Ocean Conditions
    """)
    return


@app.cell
def _(ocean_data, plt):
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

    # Wave height distribution
    axes4[0, 0].hist(ocean_data['hs'].dropna(), bins=50, edgecolor='black')
    axes4[0, 0].set_xlabel('Significant Wave Height (m)')
    axes4[0, 0].set_ylabel('Frequency')
    axes4[0, 0].set_title('Wave Height Distribution')
    axes4[0, 0].grid(True, alpha=0.3)

    # Wave length distribution
    axes4[0, 1].hist(ocean_data['lm'].dropna(), bins=50, edgecolor='black')
    axes4[0, 1].set_xlabel('Mean Wave Length (m)')
    axes4[0, 1].set_ylabel('Frequency')
    axes4[0, 1].set_title('Wave Length Distribution')
    axes4[0, 1].grid(True, alpha=0.3)

    # Wave direction distribution
    axes4[1, 0].hist(ocean_data['dir'].dropna(), bins=36, edgecolor='black')
    axes4[1, 0].set_xlabel('Wave Direction (degrees)')
    axes4[1, 0].set_ylabel('Frequency')
    axes4[1, 0].set_title('Wave Direction Distribution')
    axes4[1, 0].grid(True, alpha=0.3)

    # Sea level distribution
    axes4[1, 1].hist(ocean_data['wlv'].dropna(), bins=50, edgecolor='black')
    axes4[1, 1].set_xlabel('Sea Surface Height (m)')
    axes4[1, 1].set_ylabel('Frequency')
    axes4[1, 1].set_title('Sea Level Distribution')
    axes4[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig4
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Weather Conditions
    """)
    return


@app.cell
def _(plt, weather_obs):
    fig5, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Temperature distribution
    axes[0, 0].hist(weather_obs['T'].dropna(), bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Temperature Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Wind speed distribution
    axes[0, 1].hist(weather_obs['Ff'].dropna(), bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Wind Speed (m/s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Wind Speed Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Pressure distribution
    axes[1, 0].hist(weather_obs['P'].dropna(), bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Atmospheric Pressure (mmHg)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Pressure Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Visibility distribution
    axes[1, 1].hist(weather_obs['VV'].dropna(), bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Visibility (km)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Visibility Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig5
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Data Quality Assessment

    Check for missing values and data quality issues.
    """)
    return


@app.cell
def _(ais_dynamic, mo, pd):
    missing_ais = ais_dynamic.isnull().sum()
    missing_pct_ais = (missing_ais / len(ais_dynamic) * 100).round(2)

    quality_df_ais = pd.DataFrame({
        'Missing Count': missing_ais,
        'Missing %': missing_pct_ais
    })

    mo.md("### AIS Data Quality")
    return (quality_df_ais,)


@app.cell
def _(quality_df_ais):
    quality_df_ais
    return


@app.cell
def _(mo, ocean_data, pd):
    missing_ocean = ocean_data.isnull().sum()
    missing_pct_ocean = (missing_ocean / len(ocean_data) * 100).round(2)

    quality_df_ocean = pd.DataFrame({
        'Missing Count': missing_ocean,
        'Missing %': missing_pct_ocean
    })

    mo.md("### Ocean Data Quality")
    return (quality_df_ocean,)


@app.cell
def _(quality_df_ocean):
    quality_df_ocean
    return


@app.cell
def _(mo, pd, weather_obs):
    missing_weather = weather_obs.isnull().sum()
    missing_pct_weather = (missing_weather / len(weather_obs) * 100).round(2)

    quality_df_weather = pd.DataFrame({
        'Missing Count': missing_weather,
        'Missing %': missing_pct_weather
    })

    mo.md("### Weather Data Quality")
    return (quality_df_weather,)


@app.cell
def _(quality_df_weather):
    quality_df_weather
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Key Insights

    ### Data Coverage
    - All three datasets cover the same 6-month period (Oct 2015 - Mar 2016)
    - Geographic coverage: Longitude -10° to 0°, Latitude 45° to 51° (Brittany region)
    - AIS data provides high-resolution vessel tracking
    - Ocean data provides gridded sea state forecasts
    - Weather data from 16 coastal stations

    ### Next Steps
    1. **Scene Generation**: Create maritime scenes centered on vessels with 15-30 min windows
    2. **Feature Engineering**: Extract relevant features from all data sources
    3. **SSL Model**: Build self-supervised learning model for scene encoding
    4. **Clustering**: Apply HDBSCAN or similar to identify maritime activity patterns
    5. **Evaluation**: Assess clusters against operational activities
    """)
    return


if __name__ == "__main__":
    app.run()
