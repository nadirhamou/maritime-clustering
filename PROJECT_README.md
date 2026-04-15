# Maritime Scene Clustering - AI Model for Maritime Activity Detection

## Project Overview

This project implements a Self-Supervised Learning (SSL) approach to identify and cluster maritime activities from AIS (Automatic Identification System), ocean, and weather data. The AI model learns to encode maritime scenes into high-dimensional vectors and clusters them to identify operational activities such as:

- Port operations (entry/exit)
- Bunkering activities
- Vessel meetings
- Smuggling or suspicious activities
- Regular transit routes
- High-density shipping lanes

## Dataset

The project uses maritime data from the Brittany region (France) covering 6 months (October 2015 - March 2016):

### AIS Data
- **Source**: Naval Academy, France
- **Coverage**: Longitude -10° to 0°, Latitude 45° to 51°
- **Records**: ~18.6M dynamic messages, ~1M static messages
- **Features**: Position, speed, course, heading, vessel type, dimensions

### Ocean Data
- **Source**: SHOM - IFREMER (WAVEWATCH III model)
- **Features**: Wave height, wave length, wave direction, sea level
- **Temporal Resolution**: 3-hour forecasts

### Weather Data
- **Source**: Met Office (UK) / NOAA (US)
- **Stations**: 16 coastal weather stations
- **Features**: Temperature, pressure, humidity, wind speed/direction, visibility
- **Temporal Resolution**: Hourly observations

## Project Structure

```
.
├── 01_data_exploration.py          # Data exploration and visualization
├── 02_data_preprocessing.py        # Data cleaning and preprocessing
├── 03_scene_generation.py          # Maritime scene generation
├── 04_ssl_model_training.py        # SSL model training
├── 05_clustering_evaluation.py     # Clustering and evaluation
├── 06_inference.py                 # Inference pipeline
├── AIS_DATA/                       # AIS vessel tracking data
├── OCEAN_DATA/                     # Ocean state forecasts
├── WEATHER_DATA/                   # Weather observations
└── PROJECT_README.md               # This file
```

## Methodology

### 1. Scene Generation
Maritime scenes are created by:
- Selecting a vessel as the scene center
- Extracting all vessels within a spatial radius (5-10 nautical miles)
- Capturing data within a time window (15-30 minutes)
- Aggregating ocean and weather conditions
- Computing scene-level features

### 2. Self-Supervised Learning
An autoencoder architecture learns to:
- Encode maritime scenes into 64-dimensional embeddings
- Reconstruct scene features from embeddings
- Capture patterns and similarities between scenes
- Generate meaningful representations without labels

### 3. Clustering
HDBSCAN clustering is applied to:
- Group similar maritime scenes
- Identify distinct operational activities
- Handle noise and outliers
- Provide cluster membership probabilities

### 4. Activity Identification
Heuristic rules identify activities based on:
- Number of vessels in scene
- Average speed and speed variance
- Vessel density
- Ocean and weather conditions

## Installation

### Requirements
```bash
pip install marimo pandas numpy torch scikit-learn matplotlib seaborn
pip install umap-learn hdbscan scipy
```

### Running the Notebooks

Each notebook is a marimo file that can be run interactively:

```bash
# 1. Explore the data
marimo edit 01_data_exploration.py

# 2. Preprocess the data
marimo edit 02_data_preprocessing.py

# 3. Generate maritime scenes
marimo edit 03_scene_generation.py

# 4. Train SSL model
marimo edit 04_ssl_model_training.py

# 5. Perform clustering
marimo edit 05_clustering_evaluation.py

# 6. Run inference
marimo edit 06_inference.py
```

Or run as scripts:
```bash
marimo run 01_data_exploration.py
```

## Pipeline Workflow

### Step 1: Data Exploration
- Load and visualize AIS, ocean, and weather data
- Analyze spatial and temporal distributions
- Assess data quality and coverage

**Output**: Understanding of data characteristics

### Step 2: Data Preprocessing
- Clean and normalize data
- Handle missing values and outliers
- Create spatial indices (KD-trees) for efficient queries
- Engineer features (velocity, acceleration)

**Output**: 
- `preprocessed_ais.pkl`
- `preprocessed_ocean.pkl`
- `preprocessed_weather.pkl`
- `spatial_indices.pkl`

### Step 3: Scene Generation
- Sample vessels as scene centers
- Extract nearby vessels within time/space windows
- Aggregate ocean and weather data
- Compute scene-level features

**Output**:
- `maritime_scenes.pkl` (~1000 scenes)
- `maritime_scenes.csv`

### Step 4: SSL Model Training
- Build autoencoder architecture
- Train on scene reconstruction task
- Generate embeddings for all scenes
- Evaluate embedding quality

**Output**:
- `best_maritime_encoder.pt` (trained model)
- `scene_embeddings.npy` (64D embeddings)
- `scene_scaler.pkl` (feature scaler)
- `model_info.pkl` (configuration)

### Step 5: Clustering & Evaluation
- Apply UMAP dimensionality reduction
- Perform HDBSCAN clustering
- Evaluate clustering quality
- Identify maritime activities

**Output**:
- `umap_reducer_10d.pkl` (UMAP for clustering)
- `umap_reducer_2d.pkl` (UMAP for visualization)
- `hdbscan_clusterer.pkl` (trained clusterer)
- `cluster_labels.npy` (cluster assignments)
- `scenes_with_clusters.pkl`
- `cluster_activities.csv`

### Step 6: Inference
- Load trained models
- Process new maritime scenes
- Predict cluster assignments
- Generate activity alerts

**Output**:
- `inference_pipeline.pkl` (complete pipeline)

## Key Success Factors

### 1. Operational Activity Identification ✓
Clusters show distinct patterns linked to maritime activities:
- Port/Anchorage areas (low speed, high density)
- Bunkering/Meeting points (multiple vessels, low speed)
- Transit routes (high speed, low density)
- High-density shipping lanes

### 2. Cluster Coherence ✓
Similar activities group together:
- Limited number of clusters per activity type
- Clear separation between different activities
- Meaningful cluster characteristics

### 3. Noise Management ✓
HDBSCAN effectively handles outliers:
- Identifies noise points that don't fit patterns
- Focuses on well-defined maritime activities
- Reduces false positives

### 4. Clustering Quality Metrics
- **Silhouette Score**: Measures cluster separation
- **Davies-Bouldin Index**: Evaluates cluster compactness
- **Calinski-Harabasz Index**: Assesses cluster definition

## Model Architecture

### Autoencoder
```
Input (23 features) 
  → Dense(128) + BatchNorm + ReLU + Dropout
  → Dense(256) + BatchNorm + ReLU + Dropout
  → Dense(128) + BatchNorm + ReLU + Dropout
  → Dense(64) [Embedding Layer]
  → Dense(128) + BatchNorm + ReLU + Dropout
  → Dense(256) + BatchNorm + ReLU + Dropout
  → Dense(128) + BatchNorm + ReLU + Dropout
  → Dense(23) [Reconstruction]
```

### Training
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Epochs**: 100
- **Batch Size**: 64

## Scene Features

### Vessel Features (11)
- Number of vessels
- Number of unique vessels
- Speed statistics (mean, std, max, min)
- Course standard deviation
- Distance from center (mean, max)
- Vessel dimensions (length, beam)
- Vessel density

### Ocean Features (5)
- Significant wave height (mean, std)
- Mean wave length
- Wave direction
- Sea surface height

### Weather Features (6)
- Temperature
- Atmospheric pressure
- Humidity
- Wind speed
- Visibility
- Distance to nearest weather station

## Deployment

### Batch Processing
```python
import pickle
import pandas as pd

# Load pipeline
with open('inference_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load activity mapping
activity_dict = pd.read_csv('cluster_activities.csv')
activity_dict = dict(zip(activity_dict['Cluster'], activity_dict['Activity Type']))

# Process scene
scene_features = {...}  # Your scene features
result = pipeline.predict(scene_features)

# Get activity
activity = activity_dict.get(result['cluster'], 'Unknown')
print(f"Detected: {activity} (confidence: {result['cluster_strength']:.2%})")
```

### Real-time Monitoring
Integrate with live AIS feeds:
1. Collect AIS messages in real-time
2. Generate scenes every N minutes
3. Run inference on new scenes
4. Trigger alerts for suspicious activities

### API Service
Deploy as REST API:
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    scene = request.json
    result = pipeline.predict(scene)
    return jsonify({
        'cluster': int(result['cluster']),
        'activity': activity_dict.get(result['cluster'], 'Unknown'),
        'confidence': float(result['cluster_strength'])
    })
```

## Performance Considerations

### Computational Requirements
- **Training**: GPU recommended (CUDA-compatible)
- **Inference**: CPU sufficient for real-time processing
- **Memory**: ~4GB RAM for full pipeline

### Scalability
- Scene generation: O(n log n) with spatial indices
- Embedding generation: O(n) linear with number of scenes
- Clustering: O(n log n) with HDBSCAN

## Future Enhancements

1. **Specialized Models**: Train classifiers for specific activities (smuggling, illegal fishing)
2. **Temporal Patterns**: Incorporate time-series analysis for trajectory prediction
3. **Multi-modal Learning**: Add satellite imagery, radar data
4. **Active Learning**: Incorporate operator feedback for continuous improvement
5. **Anomaly Detection**: Identify unusual patterns not seen during training
6. **Explainability**: Add SHAP or LIME for model interpretability

## References

### Dataset
- Pallotta, G., Vespe, M., & Bryan, K. (2013). Vessel Pattern Knowledge Discovery from AIS Data: A Framework for Anomaly Detection and Route Prediction. Entropy, 15(6), 2218-2245.
- Dataset: https://zenodo.org/records/1167595

### Methods
- UMAP: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- HDBSCAN: McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering.

## License

This project uses data licensed under CC-BY-NC-SA-4.0. Please cite the original data sources when using this work.

## Contact

For questions or collaboration opportunities, please refer to the project documentation.

---

**Note**: The marimo notebooks are interactive and provide rich visualizations. Run them in marimo's edit mode for the best experience.