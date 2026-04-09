# Maritime Scene Clustering — v0.1 (Pipeline Proof of Concept)

> ** This is version 0.1 — a first end-to-end pipeline demonstration.**
> The goal of this release is to validate the full workflow from raw AIS data to interpretable clusters.
> Results, architecture, hyperparameters, and feature engineering will all evolve significantly in future versions.

---

## Overview

This project implements an **unsupervised maritime activity detection system** based on Self-Supervised Learning (SSL) and density-based clustering. Given raw AIS (Automatic Identification System) vessel tracking data, the pipeline learns to group similar maritime situations together — without any human-provided labels.

The system is designed to detect operationally relevant maritime activities such as:
- Bunkering / raft-up operations
- Port entry / exit manoeuvres
- Vessels at anchor or moored
- Isolated or suspicious vessel behaviour
- Abnormal course changes

This work is part of a broader research project aligned with maritime surveillance use cases described in the IBM datACRON framework.

---

## Dataset

**Source:** [datACRON AIS Brest Dataset — Zenodo](https://zenodo.org/record/1167595)

| File | Size | Description |
|------|------|-------------|
| `nari_dynamic.csv` | ~1.8 GB | AIS kinematic messages (position, speed, heading, navstatus) |
| `nari_static.csv` | — | Vessel static info (shiptype, dimensions, draught) |
| `ais_brest_synopses.csv` | 733 MB | Pre-computed trajectory synopses (critical points only) |
| `ais_brest_notifications.csv` | 14 MB | AIS communication gaps |

**Coverage:** Atlantic Ocean, Brest area, France — October 2015 to March 2016
**Raw records:** 18,102,412 dynamic AIS messages — 4,842 unique vessels

> The JSON files from the Zenodo archive are not needed. Only the CSV files are used.

---

## Pipeline

```
Raw AIS CSV
    │
    ▼
1. Data Loading & Filtering
   ├── Chunk-based reading (500k rows at a time)
   ├── Bounding box filter: lon ∈ [-10, 0], lat ∈ [45, 51]
   └── Timestamp validation & Unix conversion

    │
    ▼
2. Enrichment
   ├── Merge static vessel info (shiptype, length, draught)
   ├── Haversine distance function (nautical miles)
   └── NumPy array conversion for fast lookup

    │
    ▼
3. Maritime Scene Construction
   ├── Resample each vessel trajectory → pivot every 30 min (331,319 pivots)
   ├── For each pivot: extract all vessels within 7.5 nm and ±20 min
   ├── Spatial pre-filter (bounding box) + exact haversine filter
   └── 19,556 scenes built (97.8% success rate)

    │
    ▼
4. Feature Engineering
   ├── Pivot vessel features: speed, navstatus, ROT, shiptype, length, draught
   ├── Neighbourhood features: n_vessels, mean/std speed, min distance, frac_slow
   ├── Cyclic encoding of course & heading (sin/cos)
   └── RobustScaler normalisation — 16 features total

    │
    ▼
5. Self-Supervised Learning — SimCLR
   ├── Augmentation: Gaussian noise (σ=0.05) + random feature masking (10%)
   ├── Encoder: Linear(16→128→64→32) + BatchNorm + GELU
   ├── Projection head: Linear(32→64→64) — used only during training
   ├── Loss: NT-Xent (temperature=0.1)
   ├── Optimiser: Adam + CosineAnnealingLR + gradient clipping (max_norm=1.0)
   ├── 100 epochs total (50 + 50 with early stopping)
   └── Output: 32-dimensional scene embeddings

    │
    ▼
6. Dimensionality Reduction + Clustering
   ├── UMAP: 32D → 10D (cosine metric, min_dist=0.0, n_neighbors=30)
   ├── HDBSCAN grid search over min_cluster_size ∈ [20,30,40,50]
   │   and min_samples ∈ [5,10,15]
   ├── Best config: min_cluster_size=40, min_samples=15
   └── Confidence filter: keep points with HDBSCAN probability > 0.3

    │
    ▼
7. Cluster Interpretation
   ├── Per-cluster median feature profile
   ├── Rule-based heuristic labelling (AIS navstatus + behavioural thresholds)
   └── Geographic visualisation of suspicious activities
```

---

## Results (v0.1)

| Metric | Value |
|--------|-------|
| Scenes built | 19,556 |
| Clusters found | 95 |
| Noise points | 2,044 (10.5%) |
| Silhouette score | **0.621** |
| Confident points | 17,416 / 19,556 |
| Scenes flagged as suspicious | 8,815 (45.1%) |

### Top clusters detected

| Cluster | Size | Label |
|---------|------|-------|
| 1 | 2,619 | Raft-up / bunkering candidate |
| 24 | 933 | Standard transit |
| 63 | 744 | Moored (navstatus=5) |
| 62 | 596 | Moored (navstatus=5) |
| 41 | 512 | Restricted manoeuvrability |
| 26 | 502 | Raft-up / bunkering candidate |

> **Silhouette score interpretation:** < 0.3 = poor, 0.3–0.5 = acceptable, > 0.5 = good, > 0.7 = strong.
> A score of 0.621 at this stage — with no labels, no domain tuning, and a minimal feature set — is a solid baseline.

---

## Project Structure

```
maritime-clustering/
│
├── data/                          # AIS CSV files (not committed — see below)
│   ├── nari_dynamic.csv
│   ├── nari_static.csv
│   └── ...
│
├── Clustering.ipynb               # Main notebook — full pipeline
│
├── saved_models/                  # Model checkpoints
│   ├── maritime_encoder.pt        # Trained encoder weights
│   ├── best_ssl_model.pt          # Best checkpoint (early stopping)
│   ├── scaler.pkl                 # RobustScaler
│   ├── umap_reducer_10d.pkl       # UMAP reducer (10D)
│   └── hdbscan_clusterer.pkl      # HDBSCAN clusterer
│
├── df_scenes.pkl                  # Computed scene DataFrame (19,556 scenes)
├── scene_labels.npy               # HDBSCAN cluster labels
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/maritime-clustering.git
cd maritime-clustering

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
pandas>=2.0
numpy>=1.24
torch>=2.0
scikit-learn>=1.3
hdbscan>=0.8.33
umap-learn>=0.5.5
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
joblib>=1.3
scipy>=1.11
```

---

## Usage

Download the AIS Brest dataset from [Zenodo](https://zenodo.org/record/1167595) and place the CSV files in the `data/` folder, then open and run `Clustering.ipynb` cell by cell from top to bottom.

To skip retraining and load the saved model directly:

```python
import torch, joblib, numpy as np

# Load encoder
encoder.load_state_dict(
    torch.load("saved_models/best_ssl_model.pt")["encoder"]
)
encoder.eval()

# Load pipeline objects
scaler    = joblib.load("saved_models/scaler.pkl")
reducer   = joblib.load("saved_models/umap_reducer_10d.pkl")
clusterer = joblib.load("saved_models/hdbscan_clusterer.pkl")
labels    = np.load("scene_labels.npy")
```

---

## Known Limitations (v0.1)

This is a first proof-of-concept. The following aspects are known to be incomplete or improvable:

- **No external data integrated yet.** Weather, sea state, port proximity, and AIS communication gaps (notifications) are computed but not yet used as features.
- **Scene construction is sequential.** The pivot loop takes ~1 hour on CPU. A vectorised or GPU-accelerated version is planned.
- **Heuristic labelling is approximate.** The rule-based cluster labels are a starting point for human expert review, not ground truth.
- **No geographic context.** Port coordinates, shoreline proximity, and maritime separation lines are not yet factored in.
- **Encoder architecture is minimal.** The 16-feature → 32-dim encoder will be replaced by a more expressive architecture (e.g. temporal, multi-modal) in later versions.
- **Single geographic zone.** Only the Brest area is covered. Generalisation to other zones is not validated.
- **No inference pipeline.** There is currently no standalone script to classify new incoming AIS scenes against the learned clusters.

---

## Roadmap

```
v0.1  ✅  End-to-end pipeline — SSL + HDBSCAN — Brest dataset
v0.2  [ ]  Add weather, port proximity, and gap features
v0.3  [ ]  Temporal encoder (LSTM or Transformer) on vessel trajectories
v0.4  [ ]  Active learning loop — expert labels on top clusters
v0.5  [ ]  Inference API — score new scenes in real time
v1.0  [ ]  Multi-zone generalisation + full evaluation benchmark
```

---

## References

- **datACRON project** — Cross-streaming real-time detection of moving object trajectories, Deliverable D2.1, 2017
- **SimCLR** — Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020
- **HDBSCAN** — Campello et al., *Density-Based Clustering Based on Hierarchical Density Estimates*, PAKDD 2013
- **UMAP** — McInnes et al., *UMAP: Uniform Manifold Approximation and Projection*, 2018
- **AIS Brest dataset** — NARI / Institut de Recherche de l'École Navale, via datACRON Zenodo archive

---


---

*Built as part of a maritime surveillance research project. Feedback and contributions welcome.*
