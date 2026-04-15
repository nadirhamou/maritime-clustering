import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import pickle
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    mo.md("""
    # Maritime Scene Inference

    This notebook demonstrates how to use the trained SSL model for inference:
    1. Load trained models and preprocessors
    2. Process new maritime data
    3. Generate scene embeddings
    4. Predict cluster assignments
    5. Identify maritime activities
    6. Visualize results
    """)
    return mo, nn, np, pd, pickle, plt, torch


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Trained Models and Artifacts
    """)
    return


@app.cell
def _(pickle):
    print("Loading trained models and artifacts...")

    # Load feature scaler
    with open('scene_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load model configuration
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)

    # Load UMAP reducers
    with open('umap_reducer_10d.pkl', 'rb') as f:
        umap_reducer_10d = pickle.load(f)

    with open('umap_reducer_2d.pkl', 'rb') as f:
        umap_reducer_2d = pickle.load(f)

    # Load HDBSCAN clusterer
    with open('hdbscan_clusterer.pkl', 'rb') as f:
        clusterer = pickle.load(f)

    print("✓ All artifacts loaded successfully")
    return clusterer, model_info, scaler, umap_reducer_10d, umap_reducer_2d


@app.cell
def _(mo):
    mo.md("""
    ## 2. Reconstruct Model Architecture
    """)
    return


@app.cell
def _(model_info, nn, torch):
    class MaritimeSceneEncoder(nn.Module):
        """Autoencoder for learning maritime scene embeddings"""

        def __init__(self, input_dim, embedding_dim, hidden_dims, dropout_rate=0.2):
            super(MaritimeSceneEncoder, self).__init__()

            self.input_dim = input_dim
            self.embedding_dim = embedding_dim

            # Encoder
            encoder_layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim

            encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder
            decoder_layers = []
            prev_dim = embedding_dim

            for hidden_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim

            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x):
            """Generate embedding"""
            return self.encoder(x)

        def decode(self, z):
            """Reconstruct from embedding"""
            return self.decoder(z)

        def forward(self, x):
            """Full forward pass"""
            z = self.encode(x)
            x_reconstructed = self.decode(z)
            return x_reconstructed, z

    # Initialize model
    inference_model = MaritimeSceneEncoder(
        input_dim=model_info['input_dim'],
        embedding_dim=model_info['config']['embedding_dim'],
        hidden_dims=model_info['config']['hidden_dims'],
        dropout_rate=model_info['config']['dropout_rate']
    )

    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_model.load_state_dict(torch.load('best_maritime_encoder.pt', map_location=device))
    inference_model.to(device)  # in-place, pas de réassignation
    inference_model.eval()
    print(f"✓ Model loaded on {device}")
    return device, inference_model


@app.cell
def _(mo):
    mo.md("""
    ## 3. Define Inference Pipeline
    """)
    return


@app.cell
def _(
    clusterer,
    device,
    inference_model,
    model_info,
    np,
    scaler,
    torch,
    umap_reducer_10d,
    umap_reducer_2d,
):
    class MaritimeInferencePipeline:
        def __init__(self, model, scaler, umap_10d, umap_2d, clusterer, device, feature_cols):
            self.model = model
            self.scaler = scaler
            self.umap_10d = umap_10d
            self.umap_2d = umap_2d
            self.clusterer = clusterer
            self.device = device
            self.feature_cols = feature_cols

        def preprocess_scene(self, scene_features):
            features = np.array([scene_features.get(col, 0.0) for col in self.feature_cols])
            features = np.nan_to_num(features, nan=0.0)
            return self.scaler.transform(features.reshape(1, -1))

        def generate_embedding(self, features_scaled):
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                return self.model.encode(features_tensor).cpu().numpy()

        def predict_cluster(self, embedding):
            """Predict cluster via k-NN majority vote in training embeddings"""
            from sklearn.metrics.pairwise import euclidean_distances
            embedding_10d = self.umap_10d.transform(embedding)
            labels = self.clusterer.labels_
            training_embeddings = self.umap_10d.embedding_
            distances = euclidean_distances(embedding_10d, training_embeddings)[0]
            k = 5
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = labels[nearest_indices]
            valid_labels = nearest_labels[nearest_labels != -1]
            if len(valid_labels) == 0:
                return -1, 0.0
            unique, counts = np.unique(valid_labels, return_counts=True)
            cluster_label = unique[np.argmax(counts)]
            strength = float(np.max(counts) / k)
            return int(cluster_label), strength

        def get_2d_projection(self, embedding):
            return self.umap_2d.transform(embedding)

        def predict(self, scene_features):
            features_scaled = self.preprocess_scene(scene_features)
            embedding = self.generate_embedding(features_scaled)
            cluster_label, strength = self.predict_cluster(embedding)
            projection_2d = self.get_2d_projection(embedding)
            return {
                'embedding': embedding,
                'cluster': cluster_label,
                'cluster_strength': strength,
                'projection_2d': projection_2d
            }

    # Initialize pipeline
    pipeline = MaritimeInferencePipeline(
        model=inference_model,
        scaler=scaler,
        umap_10d=umap_reducer_10d,
        umap_2d=umap_reducer_2d,
        clusterer=clusterer,
        device=device,
        feature_cols=model_info['feature_cols']
    )
    print("✓ Inference pipeline initialized")
    return (pipeline,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Load Activity Type Mapping
    """)
    return


@app.cell
def _(pd):
    # Load cluster activity mapping
    try:
        activity_mapping = pd.read_csv('cluster_activities.csv')
        activity_dict = dict(zip(activity_mapping['Cluster'], activity_mapping['Activity Type']))
        print("✓ Activity mapping loaded")
    except:
        print("⚠ Activity mapping not found, using generic labels")
        activity_dict = {}
    return (activity_dict,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Example: Inference on New Scene
    """)
    return


@app.cell
def _():
    # Example new scene features
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

    print("Example scene features:")
    for key, value in new_scene.items():
        print(f"  {key}: {value}")
    return (new_scene,)


@app.cell
def _(new_scene, pipeline):
    print("\nRunning inference...")
    inference_result = pipeline.predict(new_scene)
    print(f"\n✓ Inference complete!")
    print(f"  Predicted Cluster: {inference_result['cluster']}")
    print(f"  Cluster Strength: {inference_result['cluster_strength']:.4f}")
    print(f"  Embedding shape: {inference_result['embedding'].shape}")
    print(f"  2D Projection: {inference_result['projection_2d'][0]}")
    return


@app.cell
def _(activity_dict, result):
    # Get activity type
    cluster_id = result['cluster']
    activity_type = activity_dict.get(cluster_id, "Unknown Activity")

    print(f"\n📍 Identified Activity: {activity_type}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Batch Inference on Test Scenes
    """)
    return


@app.cell
def _(pd):
    # Load some test scenes
    print("Loading test scenes...")
    scenes_df = pd.read_pickle('maritime_scenes.pkl')

    # Sample 100 random scenes for testing
    test_scenes = scenes_df.sample(n=min(100, len(scenes_df)), random_state=42)
    print(f"✓ Loaded {len(test_scenes)} test scenes")
    return (test_scenes,)


@app.cell
def _(model_info, pd, pipeline, test_scenes):
    # Run batch inference
    print("Running batch inference...")

    predictions = []
    for idx, scene in test_scenes.iterrows():
        # Extract features
        scene_features = {col: scene[col] for col in model_info['feature_cols']}

        # Predict
        result = pipeline.predict(scene_features)

        predictions.append({
            'scene_id': scene['scene_id'],
            'predicted_cluster': result['cluster'],
            'cluster_strength': result['cluster_strength'],
            'actual_cluster': scene.get('cluster', -1)
        })

    predictions_df = pd.DataFrame(predictions)
    print(f"✓ Batch inference complete on {len(predictions_df)} scenes")
    return predictions_df, result


@app.cell
def _(predictions_df):
    predictions_df.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Evaluate Inference Accuracy
    """)
    return


@app.cell
def _(predictions_df):
    # Calculate accuracy (if actual clusters are available)
    if 'actual_cluster' in predictions_df.columns:
        accuracy = (predictions_df['predicted_cluster'] == predictions_df['actual_cluster']).mean()
        print(f"Inference Accuracy: {accuracy:.2%}")

        # Confusion analysis
        correct = predictions_df[predictions_df['predicted_cluster'] == predictions_df['actual_cluster']]
        incorrect = predictions_df[predictions_df['predicted_cluster'] != predictions_df['actual_cluster']]

        print(f"Correct predictions: {len(correct)}")
        print(f"Incorrect predictions: {len(incorrect)}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Visualize Inference Results
    """)
    return


@app.cell
def _(plt, predictions_df):
    # Distribution of predicted clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted cluster distribution
    cluster_counts = predictions_df['predicted_cluster'].value_counts().sort_index()
    ax1.bar(cluster_counts.index, cluster_counts.values)
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Scenes')
    ax1.set_title('Distribution of Predicted Clusters')
    ax1.grid(True, alpha=0.3)

    # Cluster strength distribution
    ax2.hist(predictions_df['cluster_strength'], bins=30, edgecolor='black')
    ax2.set_xlabel('Cluster Strength')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Cluster Assignment Strength')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Real-time Monitoring Dashboard Concept
    """)
    return


@app.cell
def _(activity_dict, mo):
    def create_alert_message(scene_features, prediction_result):
        """Create alert message for maritime activity"""

        cluster_id = prediction_result['cluster']
        activity = activity_dict.get(cluster_id, "Unknown Activity")
        strength = prediction_result['cluster_strength']

        # Determine alert level
        if cluster_id == -1:
            alert_level = "INFO"
            message = "Unclassified maritime activity detected"
        elif "Suspicious" in activity or "Smuggling" in activity:
            alert_level = "HIGH"
            message = f"⚠️ HIGH PRIORITY: {activity} detected"
        elif "Bunkering" in activity or "Meeting" in activity:
            alert_level = "MEDIUM"
            message = f"⚡ MEDIUM PRIORITY: {activity} detected"
        else:
            alert_level = "LOW"
            message = f"ℹ️ {activity} detected"

        return {
            'alert_level': alert_level,
            'message': message,
            'activity_type': activity,
            'cluster_id': cluster_id,
            'confidence': strength,
            'num_vessels': scene_features.get('num_vessels', 0),
            'avg_speed': scene_features.get('speed_mean', 0),
            'location': (scene_features.get('center_lon', 0), scene_features.get('center_lat', 0))
        }

    mo.md("### Alert System Function Defined")
    return (create_alert_message,)


@app.cell
def _(create_alert_message, new_scene, result):
    # Example alert
    alert = create_alert_message(new_scene, result)

    print("\n" + "="*60)
    print(f"MARITIME SURVEILLANCE ALERT")
    print("="*60)
    print(f"Alert Level: {alert['alert_level']}")
    print(f"Message: {alert['message']}")
    print(f"Activity Type: {alert['activity_type']}")
    print(f"Confidence: {alert['confidence']:.2%}")
    print(f"Number of Vessels: {alert['num_vessels']}")
    print(f"Average Speed: {alert['avg_speed']:.1f} knots")
    print("="*60)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Export Inference Function
    """)
    return


@app.cell
def _(pickle, pipeline):
    def save_inference_pipeline(pipeline, filename='inference_pipeline.pkl'):
        """Save complete inference pipeline for deployment"""
        with open(filename, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"✓ Inference pipeline saved to {filename}")

    # Save pipeline
    save_inference_pipeline(pipeline)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Summary

    ### Inference System Complete

    **Capabilities:**
    1. ✓ Load trained SSL model and preprocessors
    2. ✓ Process new maritime scene data
    3. ✓ Generate scene embeddings
    4. ✓ Predict cluster assignments with confidence scores
    5. ✓ Identify maritime activity types
    6. ✓ Create alerts for suspicious activities

    **Output Files:**
    - `inference_pipeline.pkl`: Complete inference pipeline for deployment

    **Deployment Options:**

    1. **Batch Processing**: Process historical data for analysis
    2. **Real-time Monitoring**: Integrate with live AIS feeds
    3. **Alert System**: Trigger alerts for high-priority activities
    4. **API Service**: Deploy as REST API for maritime surveillance systems

    **Usage Example:**
    ```python
    # Load pipeline
    with open('inference_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    # Prepare scene features
    scene = {
        'num_vessels': 4,
        'speed_mean': 1.5,
        # ... other features
    }

    # Run inference
    result = pipeline.predict(scene)

    # Get activity type
    activity = activity_dict.get(result['cluster'], 'Unknown')
    print(f"Detected: {activity}")
    ```

    **Next Steps:**
    1. Deploy inference system to production
    2. Integrate with maritime surveillance infrastructure
    3. Collect operational feedback for model refinement
    4. Train specialized models for high-priority activities
    5. Implement continuous learning pipeline
    """)
    return


if __name__ == "__main__":
    app.run()
