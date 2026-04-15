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
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import pickle
    import warnings
    warnings.filterwarnings('ignore')

    mo.md("""
    # Self-Supervised Learning Model Training

    This notebook implements an SSL approach for maritime scene encoding:
    1. Load and prepare scene features
    2. Build autoencoder architecture for scene embedding
    3. Train the model using reconstruction loss
    4. Generate embeddings for all scenes
    5. Evaluate embedding quality
    """)
    return (
        DataLoader,
        Dataset,
        StandardScaler,
        mo,
        nn,
        np,
        optim,
        pd,
        pickle,
        plt,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Configuration
    """)
    return


@app.cell
def _(torch):
    # Model configuration
    MODEL_CONFIG = {
        'embedding_dim': 64,  # Dimension of the learned embedding
        'hidden_dims': [128, 256, 128],  # Hidden layer dimensions
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': 100,
        'dropout_rate': 0.2,
        'weight_decay': 1e-5,
    }

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nModel Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    return MODEL_CONFIG, device


@app.cell
def _(mo):
    mo.md("""
    ## 2. Load Scene Data
    """)
    return


@app.cell
def _(pd):
    print("Loading maritime scenes...")
    scenes_df = pd.read_pickle('maritime_scenes.pkl')
    print(f"✓ Loaded {len(scenes_df):,} scenes")
    return (scenes_df,)


@app.cell
def _(scenes_df):
    scenes_df.head()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Prepare Features for Training
    """)
    return


@app.cell
def _(np, scenes_df):
    # Select numerical features for training
    feature_cols = [
        'num_vessels', 'num_unique_vessels',
        'speed_mean', 'speed_std', 'speed_max', 'speed_min',
        'course_std', 'distance_mean', 'distance_max',
        'length_mean', 'beam_mean', 'vessel_density',
        'ocean_hs_mean', 'ocean_hs_std', 'ocean_lm_mean',
        'ocean_dir_mean', 'ocean_wlv_mean',
        'weather_temp', 'weather_pressure', 'weather_humidity',
        'weather_wind_speed', 'weather_visibility', 'weather_station_distance'
    ]

    # Extract features
    X = scenes_df[feature_cols].values

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    return X, feature_cols


@app.cell
def _(StandardScaler, X, pickle):
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for later use
    with open('scene_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("✓ Features normalized and scaler saved")
    return (X_scaled,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Create PyTorch Dataset
    """)
    return


@app.cell
def _(Dataset, torch):
    class SceneDataset(Dataset):
        """Dataset for maritime scenes"""

        def __init__(self, features):
            self.features = torch.FloatTensor(features)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]

    print("SceneDataset class defined")
    return (SceneDataset,)


@app.cell
def _(DataLoader, MODEL_CONFIG, SceneDataset, X_scaled):
    # Create dataset and dataloader
    dataset = SceneDataset(X_scaled)
    dataloader = DataLoader(
        dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )

    print(f"✓ Created dataloader with {len(dataset)} samples")
    return (dataloader,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Define Autoencoder Architecture
    """)
    return


@app.cell
def _(MODEL_CONFIG, X_scaled, nn):
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
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            x_reconstructed = self.decode(z)
            return x_reconstructed, z

    # Initialize model
    input_dim = X_scaled.shape[1]
    autoencoder = MaritimeSceneEncoder(
        input_dim=input_dim,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    )
    print(f"✓ Model initialized with {sum(p.numel() for p in autoencoder.parameters()):,} parameters")
    return (autoencoder,)


@app.cell
def _(autoencoder):
    print("\nModel Architecture:")
    print(autoencoder)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Training Setup
    """)
    return


@app.cell
def _(MODEL_CONFIG, autoencoder, device, nn, optim):
    # Move model to device (in-place, no reassignment)
    autoencoder.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    print("✓ Training setup complete")
    return criterion, optimizer, scheduler


@app.cell
def _(mo):
    mo.md("""
    ## 7. Training Loop
    """)
    return


@app.cell
def _(
    MODEL_CONFIG,
    autoencoder,
    criterion,
    dataloader,
    device,
    optimizer,
    scheduler,
    torch,
):
    def train_epoch(autoencoder, dataloader, criterion, optimizer, device):
        """Train for one epoch"""
        autoencoder.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed, embeddings = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(autoencoder, dataloader, criterion, device):
        """Validate the model"""
        autoencoder.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                reconstructed, embeddings = autoencoder(batch)
                loss = criterion(reconstructed, batch)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    # Training loop
    print("Starting training...")
    train_losses = []
    best_loss = float('inf')

    for epoch in range(MODEL_CONFIG['num_epochs']):
        train_loss = train_epoch(autoencoder, dataloader, criterion, optimizer, device)
        train_losses.append(train_loss)
        scheduler.step(train_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{MODEL_CONFIG['num_epochs']}], Loss: {train_loss:.6f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(autoencoder.state_dict(), 'best_maritime_encoder.pt')

    print(f"\n✓ Training complete! Best loss: {best_loss:.6f}")
    return (train_losses,)


@app.cell
def _(mo):
    mo.md("""
    ## 8. Visualize Training Progress
    """)
    return


@app.cell
def _(plt, train_losses):
    fig, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(train_losses, label='Training Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.set_title('Training Progress')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Generate Embeddings for All Scenes
    """)
    return


@app.cell
def _(X_scaled, autoencoder, device, torch):
    # Load best model
    autoencoder.load_state_dict(torch.load('best_maritime_encoder.pt'))
    autoencoder.eval()

    # Generate embeddings
    print("Generating embeddings for all scenes...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        embeddings = autoencoder.encode(X_tensor).cpu().numpy()

    print(f"✓ Generated embeddings with shape: {embeddings.shape}")
    return (embeddings,)


@app.cell
def _(embeddings, np):
    # Save embeddings
    np.save('scene_embeddings.npy', embeddings)
    print("✓ Saved scene_embeddings.npy")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Evaluate Embedding Quality
    """)
    return


@app.cell
def _(embeddings, np):
    # Calculate embedding statistics
    embedding_stats = {
        'mean': np.mean(embeddings, axis=0),
        'std': np.std(embeddings, axis=0),
        'min': np.min(embeddings, axis=0),
        'max': np.max(embeddings, axis=0)
    }

    print("Embedding Statistics:")
    print(f"  Mean norm: {np.linalg.norm(embedding_stats['mean']):.4f}")
    print(f"  Std norm: {np.linalg.norm(embedding_stats['std']):.4f}")
    print(f"  Min value: {embedding_stats['min'].min():.4f}")
    print(f"  Max value: {embedding_stats['max'].max():.4f}")
    return


@app.cell
def _(embeddings, plt):
    # Visualize embedding distributions
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distribution of first 4 embedding dimensions
    for i in range(4):
        row = i // 2
        col = i % 2
        axes[row, col].hist(embeddings[:, i], bins=50, edgecolor='black')
        axes[row, col].set_xlabel(f'Embedding Dimension {i+1}')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'Distribution of Dimension {i+1}')
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Dimensionality Reduction for Visualization
    """)
    return


@app.cell
def _(embeddings):
    from sklearn.decomposition import PCA

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return (embeddings_2d,)


@app.cell
def _(embeddings_2d, plt):
    # Visualize 2D embeddings
    fig3, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        alpha=0.5, s=20, c=range(len(embeddings_2d)), 
                        cmap='viridis')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Scene Embeddings (2D PCA Projection)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Scene Index')

    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    ## 12. Save Model and Artifacts
    """)
    return


@app.cell
def _(autoencoder, torch):
    torch.save(autoencoder.state_dict(), 'maritime_scene_encoder.pt')
    print("✓ Saved maritime_scene_encoder.pt")
    return


@app.cell
def _(MODEL_CONFIG, feature_cols, pickle):
    # Save model configuration
    model_info = {
        'config': MODEL_CONFIG,
        'feature_cols': feature_cols,
        'input_dim': len(feature_cols)
    }

    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    print("✓ Saved model_info.pkl")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 13. Summary

    ### SSL Model Training Complete

    **Output Files:**
    - `best_maritime_encoder.pt`: Best model weights
    - `maritime_scene_encoder.pt`: Final model weights
    - `scene_embeddings.npy`: Generated embeddings for all scenes
    - `scene_scaler.pkl`: Feature scaler for inference
    - `model_info.pkl`: Model configuration and metadata

    **Model Architecture:**
    - Autoencoder with bottleneck embedding layer
    - Trained using MSE reconstruction loss
    - Generates 64-dimensional embeddings

    **Next Steps:**
    - Apply UMAP for better 2D visualization
    - Use HDBSCAN for clustering
    - Evaluate clusters against operational activities
    - Build specialized classifiers for identified activities
    """)
    return


if __name__ == "__main__":
    app.run()
