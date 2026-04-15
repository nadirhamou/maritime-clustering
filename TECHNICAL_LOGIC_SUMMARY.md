# Technical Logic Summary - Maritime Scene Clustering Project

## Executive Summary

This document explains the technical logic and methodology implemented in the Maritime Scene Clustering project. The solution uses Self-Supervised Learning (SSL) to automatically discover patterns in maritime data and cluster similar activities without requiring labeled training data.

---

## 1. Data Preprocessing Logic

### Input Data Sources
- **AIS Data**: Vessel positions, speeds, courses, vessel characteristics
- **Ocean Data**: Wave heights, wave directions, sea state conditions
- **Weather Data**: Temperature, wind, pressure, visibility from coastal stations

### Preprocessing Steps

```
Raw Data → Cleaning → Feature Engineering → Spatial Indexing → Ready for Scene Generation
```

**Key Operations:**
1. **Data Cleaning**
   - Remove invalid positions (outside geographic bounds)
   - Handle missing values (replace sentinel values like -127, 511 with NaN)
   - Filter outliers (speeds > 102.2 knots, invalid courses)

2. **Feature Engineering**
   - Calculate velocity components: `vx = speed × cos(course)`, `vy = speed × sin(course)`
   - Compute acceleration: `Δspeed / Δtime`
   - Merge static vessel info (type, dimensions) with dynamic tracks

3. **Spatial Indexing**
   - Build KD-trees for AIS, ocean, and weather data
   - Enables O(log n) spatial queries instead of O(n²)
   - Critical for efficient scene extraction

**Why This Matters:** Clean, well-structured data ensures the model learns meaningful patterns rather than noise.

---

## 2. Scene Generation Logic

### Concept: What is a Maritime Scene?

A maritime scene is a **snapshot of maritime activity** in a specific area and time window, capturing:
- All vessels within radius R (5-10 nautical miles)
- Their behavior over time window T (15-30 minutes)
- Environmental conditions (ocean state, weather)

### Scene Generation Algorithm

```python
For each time bin (every 30 minutes):
    1. Select a vessel as scene center
    2. Define spatial window: center ± R nautical miles
    3. Define temporal window: center_time ± T/2 minutes
    4. Extract all vessels in this space-time window
    5. Aggregate ocean data in the area
    6. Get weather from nearest station
    7. Compute scene-level features
    8. Save scene if it has ≥ 2 vessels
```

### Scene Features (23 total)

**Vessel Features (11):**
- `num_vessels`: Total vessel count in scene
- `num_unique_vessels`: Distinct vessels (MMSI)
- `speed_mean, speed_std, speed_max, speed_min`: Speed statistics
- `course_std`: Course variation (indicates maneuvering)
- `distance_mean, distance_max`: Distance from scene center
- `length_mean, beam_mean`: Average vessel dimensions
- `vessel_density`: Vessels per square degree

**Ocean Features (5):**
- `ocean_hs_mean, ocean_hs_std`: Wave height statistics
- `ocean_lm_mean`: Mean wave length
- `ocean_dir_mean`: Wave direction
- `ocean_wlv_mean`: Sea surface height

**Weather Features (6):**
- `weather_temp`: Air temperature
- `weather_pressure`: Atmospheric pressure
- `weather_humidity`: Relative humidity
- `weather_wind_speed`: Wind speed
- `weather_visibility`: Horizontal visibility
- `weather_station_distance`: Distance to nearest station

**Why These Features:** They capture the "who, what, where, when" of maritime activity:
- **Who**: Number and types of vessels
- **What**: Speed patterns indicate activity type (anchored, transiting, maneuvering)
- **Where**: Density and distances show spatial patterns
- **When**: Environmental conditions provide context

---

## 3. Self-Supervised Learning (SSL) Logic

### Why SSL?

Traditional supervised learning requires labeled data:
```
Scene → [Model] → Label (e.g., "Bunkering", "Port Entry")
                    ↑
                    Requires manual labeling (expensive, time-consuming)
```

SSL learns patterns **without labels**:
```
Scene → [Encoder] → Embedding (64D vector)
                    ↑
                    Learns automatically from data structure
```

### Autoencoder Architecture

The model is an **autoencoder** - it learns to compress and reconstruct data:

```
Input (23 features)
    ↓
[ENCODER]
    Dense(128) + BatchNorm + ReLU + Dropout(0.2)
    Dense(256) + BatchNorm + ReLU + Dropout(0.2)
    Dense(128) + BatchNorm + ReLU + Dropout(0.2)
    Dense(64)  ← EMBEDDING (bottleneck)
    ↓
[DECODER]
    Dense(128) + BatchNorm + ReLU + Dropout(0.2)
    Dense(256) + BatchNorm + ReLU + Dropout(0.2)
    Dense(128) + BatchNorm + ReLU + Dropout(0.2)
    Dense(23)  ← Reconstructed features
    ↓
Output (23 features)
```

### Training Logic

**Objective:** Minimize reconstruction error
```
Loss = MSE(Input, Output) = Mean((Input - Reconstructed)²)
```

**Why This Works:**
1. To reconstruct accurately, the 64D embedding must capture essential information
2. Similar scenes produce similar embeddings (they compress similarly)
3. The embedding space naturally organizes scenes by similarity

**Training Process:**
```python
For each epoch:
    For each batch of scenes:
        1. Forward pass: Scene → Encoder → Embedding → Decoder → Reconstruction
        2. Calculate loss: MSE(Original, Reconstruction)
        3. Backward pass: Update weights to minimize loss
        4. Learning rate adjustment: Reduce if loss plateaus
```

**Hyperparameters:**
- Embedding dimension: 64 (balance between information and compression)
- Hidden layers: [128, 256, 128] (gradual compression/expansion)
- Dropout: 0.2 (prevents overfitting)
- Learning rate: 0.001 with adaptive reduction
- Batch size: 64 (stable gradients)
- Epochs: 100 (convergence)

**Key Insight:** The 64D embedding is a **learned representation** where:
- Similar maritime activities are close together
- Different activities are far apart
- The model discovers this structure automatically

---

## 4. Clustering Logic

### Why UMAP + HDBSCAN?

**UMAP (Uniform Manifold Approximation and Projection):**
- Reduces 64D embeddings to 10D for clustering
- Preserves both local and global structure
- Faster and more scalable than t-SNE

**HDBSCAN (Hierarchical Density-Based Spatial Clustering):**
- Finds clusters of varying densities
- Automatically determines number of clusters
- Identifies noise/outliers (label = -1)
- Provides cluster membership probabilities

### Clustering Pipeline

```
64D Embeddings
    ↓
[UMAP Reduction]
    10D Embeddings (for clustering)
    2D Embeddings (for visualization)
    ↓
[HDBSCAN Clustering]
    Cluster Labels (0, 1, 2, ..., -1 for noise)
    Cluster Probabilities (0.0 to 1.0)
```

### HDBSCAN Algorithm Logic

```python
1. Build minimum spanning tree of data points
2. Convert to hierarchy of clusters
3. Extract stable clusters based on:
   - min_cluster_size: Minimum points per cluster (20)
   - min_samples: Core point threshold (5)
   - cluster_selection_epsilon: Merge threshold (0.0)
4. Assign points to clusters or mark as noise
```

**Parameters Explained:**
- `min_cluster_size=20`: Clusters must have ≥20 scenes (ensures statistical significance)
- `min_samples=5`: Point needs 5 neighbors to be "core" (reduces noise sensitivity)
- `cluster_selection_epsilon=0.0`: No forced merging (let data decide)

### Cluster Interpretation Logic

After clustering, we identify activity types using heuristics:

```python
def identify_activity(cluster_features):
    if num_vessels > 5 AND density > 0.01 AND speed < 2:
        return "Port/Anchorage Area"
    
    elif num_vessels > 3 AND speed < 1 AND speed_std < 2:
        return "Bunkering/Meeting"
    
    elif speed > 15 AND speed_std < 3:
        return "High-Speed Transit"
    
    elif speed < 5 AND speed_std > 5:
        return "Variable Speed Activity"
    
    elif density > 0.015:
        return "High Density Area"
    
    else:
        return "General Maritime Activity"
```

**Logic Behind Heuristics:**
- **Port areas**: Many slow vessels in small area
- **Bunkering**: Few vessels, very slow, stable (fuel transfer)
- **Transit**: Fast, consistent speed (traveling)
- **Variable speed**: Maneuvering, searching, fishing
- **High density**: Shipping lanes, convergence zones

---

## 5. Evaluation Logic

### Clustering Quality Metrics

**1. Silhouette Score (-1 to 1)**
```
For each point:
    a = average distance to points in same cluster
    b = average distance to points in nearest other cluster
    silhouette = (b - a) / max(a, b)

Overall score = average of all points
```
- **> 0.5**: Strong clustering
- **0.3-0.5**: Reasonable clustering
- **< 0.3**: Weak clustering

**2. Davies-Bouldin Index (lower is better)**
```
For each cluster pair:
    Measure: (within-cluster scatter) / (between-cluster separation)

Score = average of worst-case pairs
```
- Penalizes clusters that are too close or too spread out

**3. Calinski-Harabasz Index (higher is better)**
```
Score = (between-cluster variance) / (within-cluster variance)
      × (n_samples - n_clusters) / (n_clusters - 1)
```
- Rewards tight, well-separated clusters

### Success Criteria

✓ **Operational Relevance**: Clusters correspond to real maritime activities
✓ **Cluster Coherence**: Similar activities in same/few clusters
✓ **Noise Management**: Outliers properly identified
✓ **Stability**: Consistent results across runs
✓ **Actionability**: Clear activity types for operators

---

## 6. Inference Logic

### Inference Pipeline

```
New Scene Features
    ↓
[1. Preprocessing]
    - Normalize using saved scaler
    - Handle missing values
    ↓
[2. Embedding Generation]
    - Pass through trained encoder
    - Get 64D embedding
    ↓
[3. Dimensionality Reduction]
    - Apply saved UMAP (64D → 10D)
    ↓
[4. Cluster Prediction]
    - Use HDBSCAN approximate_predict()
    - Get cluster label + confidence
    ↓
[5. Activity Identification]
    - Map cluster to activity type
    - Generate alert if needed
    ↓
Result: {cluster, activity, confidence}
```

### Approximate Prediction Logic

HDBSCAN's `approximate_predict()`:
```python
1. Find k nearest neighbors in training data
2. Check their cluster assignments
3. If majority belong to cluster C:
   - Assign to cluster C
   - Confidence = proportion in majority
4. Else:
   - Assign to noise (-1)
   - Low confidence
```

**Why "Approximate":** Exact prediction requires rebuilding the entire clustering hierarchy, which is computationally expensive. Approximate prediction uses k-NN for speed.

---

## 7. Key Design Decisions

### Decision 1: Why Autoencoder over Contrastive Learning?

**Autoencoder Advantages:**
- Simpler to implement and debug
- No need for positive/negative pair generation
- Works well with limited data
- Interpretable reconstruction loss

**Alternative (Contrastive Learning):**
- Requires careful pair selection
- More complex training loop
- Better for very large datasets

**Decision:** Autoencoder chosen for simplicity and effectiveness with available data.

### Decision 2: Why 64D Embeddings?

**Trade-off:**
- Too low (e.g., 16D): Loss of information
- Too high (e.g., 256D): Overfitting, slow clustering
- **64D**: Sweet spot for 23 input features

**Rule of thumb:** Embedding dim ≈ 2-3× input features

### Decision 3: Why HDBSCAN over K-Means?

**K-Means Issues:**
- Requires pre-specifying number of clusters
- Assumes spherical clusters
- Poor with noise/outliers

**HDBSCAN Advantages:**
- Automatic cluster number detection
- Handles varying densities
- Robust to noise
- Provides confidence scores

### Decision 4: Scene Window Parameters

**Time Window: 20 minutes**
- Too short (5 min): Incomplete activity capture
- Too long (60 min): Multiple activities mixed
- **20 min**: Captures single activity episode

**Spatial Radius: 7.5 nautical miles**
- Too small (2 nm): Miss related vessels
- Too large (20 nm): Include unrelated traffic
- **7.5 nm**: Typical interaction distance

---

## 8. Mathematical Foundations

### Embedding Space Properties

The learned embedding space has these properties:

**1. Distance Metric:**
```
distance(scene_A, scene_B) = ||embedding_A - embedding_B||₂
```
Small distance → Similar activities

**2. Cluster Density:**
```
density(point) = 1 / (average distance to k nearest neighbors)
```
High density → Core of activity pattern

**3. Cluster Stability:**
```
stability(cluster) = Σ (1/λ_birth - 1/λ_death) for points in cluster
```
Where λ is the distance scale at which cluster appears/disappears

### UMAP Optimization

UMAP minimizes:
```
Loss = Σ log(1 + ||v_i - v_j||²) - Σ log(1 - 1/(1 + ||v_i - v_k||²))
       ↑                              ↑
    Attract similar points      Repel dissimilar points
```

This creates a manifold that preserves:
- Local structure (nearby points stay nearby)
- Global structure (distant points stay distant)

---

## 9. Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Data → Preprocessing → Scene Generation                │
│      ↓                                                       │
│  Feature Extraction (23 features per scene)                 │
│      ↓                                                       │
│  SSL Training (Autoencoder)                                 │
│      ↓                                                       │
│  Generate Embeddings (64D)                                  │
│      ↓                                                       │
│  UMAP Reduction (64D → 10D)                                 │
│      ↓                                                       │
│  HDBSCAN Clustering                                         │
│      ↓                                                       │
│  Activity Identification                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  New Scene → Preprocessing → Encoder → Embedding            │
│                                  ↓                           │
│                            UMAP → 10D                        │
│                                  ↓                           │
│                         HDBSCAN Predict                      │
│                                  ↓                           │
│                         Activity Label                       │
│                                  ↓                           │
│                         Alert (if needed)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Code Logic Examples

### Scene Feature Extraction
```python
def extract_scene_features(center_vessel, time_window, spatial_radius):
    """
    Logic: Aggregate all relevant information around a vessel
    """
    # 1. Get nearby vessels
    vessels = get_vessels_in_radius(
        center=center_vessel.position,
        radius=spatial_radius,
        time_start=center_vessel.time - time_window/2,
        time_end=center_vessel.time + time_window/2
    )
    
    # 2. Compute vessel statistics
    features = {
        'num_vessels': len(vessels),
        'speed_mean': vessels['speed'].mean(),
        'speed_std': vessels['speed'].std(),
        'vessel_density': len(vessels) / (π × radius²)
    }
    
    # 3. Get ocean conditions
    ocean = get_ocean_data(center_vessel.position, center_vessel.time)
    features.update({
        'ocean_hs_mean': ocean['wave_height'].mean(),
        'ocean_dir_mean': ocean['wave_direction'].mean()
    })
    
    # 4. Get weather
    weather = get_nearest_weather(center_vessel.position, center_vessel.time)
    features.update({
        'weather_wind_speed': weather['wind_speed'],
        'weather_temp': weather['temperature']
    })
    
    return features
```

### SSL Training Loop
```python
def train_ssl_model(scenes, model, optimizer, epochs):
    """
    Logic: Learn to compress and reconstruct scenes
    """
    for epoch in range(epochs):
        for batch in scenes:
            # Forward pass
            embeddings = model.encode(batch)      # 23D → 64D
            reconstructed = model.decode(embeddings)  # 64D → 23D
            
            # Calculate loss
            loss = MSE(batch, reconstructed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # The model learns: similar scenes → similar embeddings
```

### Clustering Logic
```python
def cluster_scenes(embeddings):
    """
    Logic: Group similar embeddings into clusters
    """
    # 1. Reduce dimensionality
    embeddings_10d = umap.fit_transform(embeddings)  # 64D → 10D
    
    # 2. Find dense regions
    clusterer = HDBSCAN(min_cluster_size=20)
    labels = clusterer.fit_predict(embeddings_10d)
    
    # 3. Identify activity types
    for cluster_id in unique(labels):
        if cluster_id == -1:
            continue  # Skip noise
        
        cluster_scenes = scenes[labels == cluster_id]
        activity = identify_activity_type(cluster_scenes.mean())
        
    return labels, activities
```

---

## 11. Why This Approach Works

### Theoretical Foundation

**1. Manifold Hypothesis**
- High-dimensional data (maritime scenes) lies on a lower-dimensional manifold
- SSL discovers this manifold structure
- Similar activities cluster naturally on the manifold

**2. Density-Based Clustering**
- Real maritime activities form dense regions in embedding space
- Noise and outliers are sparse
- HDBSCAN exploits this density difference

**3. Transfer Learning Principle**
- Model learns general maritime patterns
- Can generalize to new, unseen scenes
- No need to retrain for each new activity type

### Practical Advantages

✓ **No Manual Labeling**: Learns from data structure
✓ **Scalable**: Handles thousands of scenes efficiently
✓ **Adaptable**: New activities discovered automatically
✓ **Interpretable**: Clusters have clear characteristics
✓ **Deployable**: Fast inference (<1 second per scene)

---

## 12. Limitations and Future Improvements

### Current Limitations

1. **Static Features**: Doesn't capture temporal evolution within scenes
2. **Heuristic Activities**: Activity identification uses simple rules
3. **Limited Context**: Doesn't consider historical vessel behavior
4. **No Anomaly Detection**: Focuses on clustering, not outlier detection

### Proposed Improvements

1. **Temporal Modeling**: Add LSTM/Transformer for trajectory analysis
2. **Supervised Fine-tuning**: Use labeled data to refine activity classification
3. **Multi-modal Learning**: Incorporate satellite imagery, radar data
4. **Online Learning**: Update model with new data continuously
5. **Explainability**: Add attention mechanisms to show why scenes cluster together

---

## Conclusion

This solution implements a complete **unsupervised maritime activity detection system** using:

1. **Scene-based representation**: Captures spatial-temporal maritime context
2. **Self-supervised learning**: Discovers patterns without labels
3. **Density-based clustering**: Groups similar activities automatically
4. **Heuristic interpretation**: Maps clusters to operational activities

The key innovation is using SSL to learn a meaningful embedding space where maritime activities naturally cluster, enabling automatic discovery of operational patterns in maritime surveillance data.

---

**For Questions or Clarifications:**
- Review the marimo notebooks for implementation details
- Check `PROJECT_README.md` for deployment guidance
- Consult `QUICKSTART.md` for hands-on examples