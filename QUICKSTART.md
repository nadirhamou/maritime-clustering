# Quick Start Guide - Maritime Scene Clustering

This guide will help you get started with the Maritime Scene Clustering project in minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended)
- GPU optional (for faster training)

## Installation

### 1. Install Dependencies

```bash
pip install marimo pandas numpy torch scikit-learn matplotlib seaborn scipy
pip install umap-learn hdbscan
```

### 2. Verify Data

Ensure you have the following data directories:
```
AIS_DATA/
  ├── nari_dynamic.csv
  ├── nari_static.csv
  └── ...

OCEAN_DATA/
  ├── oc_october.csv
  ├── oc_november.csv
  └── ...

WEATHER_DATA/
  ├── table_wheatherObservation.csv
  ├── table_weatherStation.csv
  └── ...
```

## Running the Pipeline

### Option 1: Interactive Mode (Recommended)

Run each notebook interactively with marimo:

```bash
# Step 1: Explore the data
marimo edit 01_data_exploration.py

# Step 2: Preprocess the data
marimo edit 02_data_preprocessing.py

# Step 3: Generate scenes
marimo edit 03_scene_generation.py

# Step 4: Train SSL model
marimo edit 04_ssl_model_training.py

# Step 5: Cluster and evaluate
marimo edit 05_clustering_evaluation.py

# Step 6: Run inference
marimo edit 06_inference.py
```

### Option 2: Batch Mode

Run all notebooks sequentially:

```bash
marimo run 01_data_exploration.py
marimo run 02_data_preprocessing.py
marimo run 03_scene_generation.py
marimo run 04_ssl_model_training.py
marimo run 05_clustering_evaluation.py
marimo run 06_inference.py
```

## Expected Outputs

After running the complete pipeline, you'll have:

### Preprocessed Data
- `preprocessed_ais.pkl` - Cleaned AIS data
- `preprocessed_ocean.pkl` - Cleaned ocean data
- `preprocessed_weather.pkl` - Cleaned weather data
- `spatial_indices.pkl` - Spatial search structures

### Generated Scenes
- `maritime_scenes.pkl` - Scene features
- `maritime_scenes.csv` - Human-readable scenes

### Trained Models
- `best_maritime_encoder.pt` - Trained SSL model
- `scene_embeddings.npy` - 64D embeddings
- `scene_scaler.pkl` - Feature scaler
- `model_info.pkl` - Model configuration

### Clustering Results
- `umap_reducer_10d.pkl` - UMAP for clustering
- `umap_reducer_2d.pkl` - UMAP for visualization
- `hdbscan_clusterer.pkl` - Trained clusterer
- `cluster_labels.npy` - Cluster assignments
- `scenes_with_clusters.pkl` - Scenes with clusters
- `cluster_activities.csv` - Activity types

### Inference Pipeline
- `inference_pipeline.pkl` - Complete inference system

## Quick Inference Example

Once you've run the pipeline, you can use the inference system:

```python
import pickle
import pandas as pd

# Load inference pipeline
with open('inference_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load activity mapping
activities = pd.read_csv('cluster_activities.csv')
activity_dict = dict(zip(activities['Cluster'], activities['Activity Type']))

# Define a new scene
new_scene = {
    'num_vessels': 4,
    'num_unique_vessels': 4,
    'speed_mean': 1.5,
    'speed_std': 0.8,
    'speed_max': 3.2,
    'speed_min': 0.5,
    'course_std': 15.0,
    'distance_mean': 0.05,
    'distance_max': 0.1,
    'length_mean': 150.0,
    'beam_mean': 25.0,
    'vessel_density': 0.012,
    'ocean_hs_mean': 2.5,
    'ocean_hs_std': 0.5,
    'ocean_lm_mean': 180.0,
    'ocean_dir_mean': 270.0,
    'ocean_wlv_mean': 0.5,
    'weather_temp': 12.0,
    'weather_pressure': 760.0,
    'weather_humidity': 85,
    'weather_wind_speed': 8.0,
    'weather_visibility': 10.0,
    'weather_station_distance': 0.2
}

# Run inference
result = pipeline.predict(new_scene)

# Get activity type
activity = activity_dict.get(result['cluster'], 'Unknown Activity')
confidence = result['cluster_strength']

print(f"Detected Activity: {activity}")
print(f"Confidence: {confidence:.2%}")
print(f"Cluster ID: {result['cluster']}")

```

## Typical Runtime

On a modern laptop (without GPU):
- Data Exploration: 2-3 minutes
- Data Preprocessing: 5-10 minutes
- Scene Generation: 10-15 minutes
- SSL Model Training: 20-30 minutes
- Clustering & Evaluation: 5-10 minutes
- Inference: < 1 second per scene

**Total Pipeline Runtime: ~45-70 minutes**

With GPU:
- SSL Model Training: 5-10 minutes
- **Total Pipeline Runtime: ~25-40 minutes**

## Understanding the Results

### Cluster Interpretation

The system identifies several types of maritime activities:

1. **Port/Anchorage Areas**
   - Low speed (< 2 knots)
   - High vessel density
   - Multiple vessels in small area

2. **Bunkering/Meeting Points**
   - Very low speed (< 1 knot)
   - 2-4 vessels close together
   - Low speed variance

3. **Transit Routes**
   - High speed (> 8 knots)
   - Low vessel density
   - Consistent course

4. **High-Density Shipping Lanes**
   - Multiple vessels
   - Moderate to high speeds
   - High vessel density

5. **Suspicious Activities**
   - Unusual speed patterns
   - Unexpected vessel meetings
   - Deviations from normal routes

### Evaluation Metrics

- **Silhouette Score** (0.3-0.5): Good cluster separation
- **Number of Clusters** (10-50): Reasonable activity diversity
- **Noise Percentage** (10-30%): Acceptable outlier detection

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in `04_ssl_model_training.py` (MODEL_CONFIG['batch_size'] = 32)

**Issue**: Too many/few clusters
- **Solution**: Adjust HDBSCAN parameters in `05_clustering_evaluation.py`:
  - Increase `min_cluster_size` for fewer clusters
  - Decrease `min_cluster_size` for more clusters

**Issue**: Slow scene generation
- **Solution**: Reduce number of scenes in `03_scene_generation.py` (line 118: `time_bins[:1000]` → `time_bins[:500]`)

**Issue**: Poor clustering quality
- **Solution**: 
  - Train model for more epochs
  - Adjust UMAP parameters
  - Collect more diverse scenes

## Next Steps

1. **Explore Results**: Open `05_clustering_evaluation.py` to visualize clusters
2. **Validate Activities**: Review identified activities with domain experts
3. **Fine-tune Model**: Adjust hyperparameters based on results
4. **Deploy System**: Use `inference_pipeline.pkl` for production
5. **Continuous Learning**: Collect feedback and retrain model

## Getting Help

- Review `PROJECT_README.md` for detailed documentation
- Check individual notebook comments for specific guidance
- Examine visualization outputs for insights

## Tips for Best Results

1. **Data Quality**: Ensure AIS data covers diverse maritime activities
2. **Scene Diversity**: Sample scenes from different times and locations
3. **Feature Engineering**: Add domain-specific features if available
4. **Hyperparameter Tuning**: Experiment with model architecture and clustering parameters
5. **Validation**: Always validate results with maritime domain experts

---

Happy clustering! 🚢⚓🌊