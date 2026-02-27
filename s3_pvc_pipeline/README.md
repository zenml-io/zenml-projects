# S3-to-PVC Training Pipeline with ZenML

> **TL;DR** — Use Kubernetes Persistent Volume Claims as a close-to-compute data
> cache for large, versioned datasets stored in S3. ZenML orchestrates the
> download-once-train-many pattern so every pipeline step reads from local NVMe
> instead of pulling terabytes over the network.

![Pipeline Diagram](./assets/pipeline.excalidraw.svg)

---

## The Problem

Many production ML workloads share the same data pattern:

1. **A large, slow-changing dataset** (images, embeddings, tabular parquet
   files, ...) lives in long-term object storage (S3, GCS, Azure Blob).
2. The dataset is **versioned with releases or tags** (`v1`, `v2`, `2024-Q3`,
   ...) — new versions appear infrequently (weekly/monthly), but the data itself
   is huge (tens of GB to low TB).
3. Training jobs **run often** — hyperparameter sweeps, nightly retraining,
   experimentation — and every run needs the full (or most of the) dataset.

### Why a naive approach fails

| Approach | Problem |
|---|---|
| Download from S3 every run | Network I/O dominates wall-clock time. S3 GET costs add up. Cluster egress bills explode. |
| Bake data into the Docker image | Images become huge (>50 GB). Build times skyrocket. Every version bump forces a full image rebuild + push. |
| Mount S3 via FUSE (s3fs / goofys) | Random-read latency is 10-100x worse than local disk. Training throughput collapses, especially for dataloaders with `num_workers > 0`. |

### The PVC-as-cache pattern

The idea is simple:

```
S3 (source of truth, versioned)
        │
        │  download once per version
        ▼
  Kubernetes PVC  ◄──  local NVMe / EBS gp3 / etc.
        │
        │  every pipeline run reads from here
        ▼
  Training pods (fast local I/O)
```

A **Persistent Volume Claim** survives across pod restarts and pipeline runs. On
the first run for a given data version, the pipeline pulls from S3 and extracts
into the PVC. On every subsequent run, the data is already there — the step
detects the cached files and skips the download entirely.

**Result:** only the first run pays the download cost. All following runs start
training within seconds.

---

## How ZenML Solves This

ZenML provides first-class primitives that make this pattern declarative and
reproducible:

| Concern | ZenML feature |
|---|---|
| **Attach a PVC to every step** | `KubernetesPodSettings` — declare volumes, volume mounts, and resource requests in Python. Applied uniformly to every step pod. |
| **Authenticate to S3 without baked-in credentials** | **Service Connectors** — the pipeline references a connector by name; ZenML injects short-lived credentials into the pod at runtime. |
| **Version-aware caching** | The `load_data` step writes to `/mnt/data/{data_version}/`. Switching from `v1` to `v2` is a config change — both versions can coexist on the PVC. |
| **Reproducible config** | YAML run configuration (`config/config.yaml`) pins the data version, hyperparameters, and S3 coordinates. Every run is fully specified. |
| **Metadata & visualization** | Steps log metrics via `log_metadata()` and return `HTMLString` visualizations visible in the ZenML dashboard. |
| **Portable orchestration** | The same pipeline code runs on any Kubernetes cluster — EKS, GKE, AKS, on-prem — by swapping the ZenML stack. |

### Pipeline architecture

```
                  ┌──────────────────────────────────────────────────────────────────────┐
                  │                     Kubernetes Cluster (EKS)                          │
                  │                                                                      │
  ┌─────────┐    │  ┌───────────┐    ┌────────────┐    ┌─────────────┐    ┌───────────┐ │
  │  S3      │───▶│  │ load_data │───▶│ preprocess │───▶│ train_model │───▶│ test_model │ │
  │ (source  │    │  └─────┬─────┘    └─────┬──────┘    └──────┬──────┘    └─────┬─────┘ │
  │ of truth)│    │        │                │                  │                 │        │
  └──────────┘    │        ▼                ▼                  ▼                 ▼        │
                  │  ┌─────────────────────────────────────────────────────────────────┐  │
                  │  │               PVC  /mnt/data/{version}/                         │  │
                  │  │  raw/train/  raw/test/  preprocessed/train/ val/ test/          │  │
                  │  └─────────────────────────────────────────────────────────────────┘  │
                  │                                                                      │
                  │  ┌─────────────────────────────────────────────────────────────────┐  │
                  │  │                ZenML Artifact Store (S3)                         │  │
                  │  │      model checkpoints, metrics, visualizations                 │  │
                  │  └─────────────────────────────────────────────────────────────────┘  │
                  └──────────────────────────────────────────────────────────────────────┘
```

**Steps:**

| # | Step | What it does |
|---|---|---|
| 1 | `load_data` | Connects to S3 via ZenML Service Connector, downloads `train.zip` + `test.zip` for the requested `data_version`, extracts into PVC. Skips download if cached `.npy` files already exist. |
| 2 | `preprocess` | Reads raw data from PVC, performs stratified train/val split, normalizes pixel values, writes processed splits back to PVC. Logs split sizes as metadata. |
| 3 | `train_model` | Loads processed train/val data from PVC, trains a CNN (PyTorch Lightning), logs val loss/accuracy as metadata. Returns the trained `LightningModule`. |
| 4 | `test_model` | Evaluates trained model on the test split, logs test metrics, generates a sample-predictions HTML visualization for the ZenML dashboard. |

---

## Prerequisites

### Infrastructure / Kubernetes

You need a running Kubernetes cluster with:

- [ ] **`kubectl` access** — you can run `kubectl get nodes` and see your
  cluster.
- [ ] **A StorageClass that supports `ReadWriteMany`** (or `ReadWriteOnce` if
  only one pod reads at a time). Check with:
  ```shell
  kubectl get storageclass
  ```
  Common choices: EFS-backed `efs-sc` on EKS, Filestore on GKE, or
  `azurefile` on AKS.
- [ ] **Sufficient PVC capacity** — the demo uses 10 Gi by default; for real
  datasets, size accordingly.
- [ ] **Pod resource quotas** — the pipeline requests 2 CPU / 4 Gi memory per
  step pod (configurable in `pipelines/training.py`).

### ZenML Stack

Register a ZenML stack with these components (all are required):

| Component | Flavor | Example |
|---|---|---|
| **Orchestrator** | `kubernetes` | An EKS / GKE / AKS cluster |
| **Artifact Store** | `s3` (or `gcs` / `azure`) | `s3://your-zenml-artifacts` |
| **Container Registry** | `ecr` (or `gcr` / `acr` / `dockerhub`) | Your ECR repo URI |
| **Image Builder** | `local` or `kaniko` or `aws_codebuild` | AWS CodeBuild recommended for EKS |
| **Service Connector** | `aws` (type: `s3-bucket`) | Must grant read access to the data bucket |

Verify your stack:

```shell
zenml stack describe
```

### External S3 Data Bucket

An S3 bucket holding your versioned dataset. This is **separate from the ZenML
artifact store** — it is your long-term data lake / feature store.

Expected layout:

```
s3://<bucket>/
  └── <prefix>/
      └── <version>/
          ├── train.zip
          └── test.zip
```

Each zip contains class-sharded `.npy` files:

```
train/
├── class_0/
│   ├── 00000.npy
│   └── ...
├── class_1/
│   └── ...
└── class_9/
    └── ...
```

---

## Getting Started

### 1. Upload demo data to S3

The repo includes a helper script that downloads FashionMNIST, converts it to
`.npy`, zips it, and uploads to S3:

```shell
# Dry-run (download + prepare locally, no upload)
uv run scripts/upload_mnist_to_s3.py --dry-run

# Upload to S3
AWS_PROFILE=<your-profile> uv run scripts/upload_mnist_to_s3.py --bucket persistent-data-store
```

### 2. Create the PVC on your cluster and wire it into the pipeline

The PVC name you create in Kubernetes **must match** the name the pipeline
mounts into step pods. There are three places where PVC/mount settings live:

| Setting | File | Default |
|---|---|---|
| PVC claim name | `pipelines/training.py` → `PREPROCESS_PVC_CLAIM_NAME` | `zenml-data-pvc` |
| Mount path inside pods | `pipelines/training.py` → `PREPROCESS_MOUNT_PATH` | `/mnt/data` |
| Mount path (runtime override) | `config/config.yaml` → `parameters.preprocess_mount_path` | `/mnt/data` |

**Step 2a — Pick a PVC name and check your storage class:**

```shell
kubectl get storageclass
```

**Step 2b — Generate the K8s manifest with the `./run` helper:**

```shell
# Usage: ./run create_pvc_manifest <pvc-name> <namespace> [storageClass] [accessModes] [storage]
./run create_pvc_manifest zenml-data-pvc zenml efs-sc ReadWriteMany 20Gi
```

This writes `k8s/pvc.yaml`. Review it:

```shell
cat k8s/pvc.yaml
```

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zenml-data-pvc          # ← must match PREPROCESS_PVC_CLAIM_NAME
  namespace: zenml
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: efs-sc
  resources:
    requests:
      storage: 20Gi
```

**Step 2c — Apply and verify:**

```shell
kubectl apply -f k8s/pvc.yaml
kubectl get pvc zenml-data-pvc -n zenml
```

**Step 2d — If you used a different PVC name**, update the constant in
`pipelines/training.py` so the pipeline mounts the right volume:

```python
# pipelines/training.py — change this to match your kubectl PVC name
PREPROCESS_PVC_CLAIM_NAME = "zenml-data-pvc"   # ← your PVC name here
PREPROCESS_MOUNT_PATH = "/mnt/data"             # ← change if needed
```

If you also changed the mount path, update `config/config.yaml` to match:

```yaml
parameters:
  preprocess_mount_path: "/mnt/data"            # ← must match PREPROCESS_MOUNT_PATH
```

### 3. Install dependencies and configure ZenML

```shell
# Create venv and install
uv venv --seed
source .venv/bin/activate
pip install -r requirements.txt

# Login and configure ZenML
zenml login <workspace-url>
zenml init
zenml stack set <your-k8s-stack>
zenml project set <your-project>

# Install required ZenML integrations
zenml integration install aws s3 kubernetes --uv --yes
```

### 4. Run the pipeline

```shell
# Run with default config (config/config.yaml)
python run.py

# Run with a custom config
python run.py --config path/to/custom_config.yaml
```

### 5. View results in the ZenML dashboard

After the run completes:

- **Pipeline DAG** — see the step execution graph and durations.
- **Metadata** — `preprocess` logs data split sizes; `train_model` logs val
  loss/accuracy; `test_model` logs test metrics.
- **Visualization** — the `test_model` step outputs an HTML visualization of
  sample predictions, viewable directly in the dashboard.

---

## Configuration Reference

Configuration is split between two files. **Both must agree on the mount path.**

### `config/config.yaml` — runtime parameters (change per run)

```yaml
parameters:
  service_connector_name_or_id: "aws"     # ZenML service connector name
  s3_bucket: "persistent-data-store"       # S3 bucket holding versioned data
  s3_prefix: "mnist/"                      # S3 key prefix (version is appended)
  data_version: "v1"                       # Data version tag (→ PVC subdir + S3 subprefix)
  train_ratio: 0.9                         # Train/val split ratio
  seed: 42                                 # Random seed for reproducibility
  preprocess_mount_path: "/mnt/data"       # ⚠ Must match PREPROCESS_MOUNT_PATH in training.py
  hyperparams:
    learning_rate: 0.001
    batch_size: 64
    hidden_dim: 32
    max_epochs: 1
```

### `pipelines/training.py` — infrastructure settings (change once per cluster)

```python
PREPROCESS_PVC_CLAIM_NAME = "zenml-data-pvc"   # ⚠ Must match your kubectl PVC name
PREPROCESS_MOUNT_PATH = "/mnt/data"             # ⚠ Must match config.yaml
```

These constants define the Kubernetes `volumes` / `volumeMounts` and resource
requests injected into every step pod via `KubernetesPodSettings`.

### How the names flow

```
./run create_pvc_manifest <name> ...   →   k8s/pvc.yaml (metadata.name)
                                              ↕  must match
                                       pipelines/training.py (PREPROCESS_PVC_CLAIM_NAME)

                                       pipelines/training.py (PREPROCESS_MOUNT_PATH)
                                              ↕  must match
                                       config/config.yaml (parameters.preprocess_mount_path)
```

To train on a new data version, change `data_version` in `config.yaml` (e.g.,
`v2`). The PVC stores both versions side by side under
`/mnt/data/v1/`, `/mnt/data/v2/`, etc. — no need to delete the old data.

---

## Project Structure

```
s3_pvc_pipeline/
├── README.md
├── requirements.txt
├── run                          # Bash helper (create PVC manifest, clean)
├── run.py                       # Python entry point
├── config/
│   └── config.yaml              # Pipeline run configuration
├── pipelines/
│   └── training.py              # Pipeline definition + K8s/PVC settings
├── steps/
│   ├── load_data.py             # S3 → PVC download (with caching)
│   ├── preprocess.py            # Train/val split + normalization
│   ├── train_model.py           # PyTorch Lightning training
│   ├── test_model.py            # Evaluation + HTML visualization
│   ├── dataloader.py            # NumpyDirDataset + DataLoader builder
│   ├── model.py                 # CNN architecture (LightningModule)
│   └── utils.py                 # Visualization helpers
├── scripts/
│   └── upload_mnist_to_s3.py    # One-time data upload to S3
└── assets/
    └── pipeline.excalidraw.svg  # Pipeline architecture diagram
```

---

## Adapting to Your Own Dataset

This demo uses FashionMNIST, but the pattern generalizes to any large dataset:

1. **Upload your data** to S3 in the versioned layout
   (`s3://bucket/prefix/version/train.zip`, etc.).
2. **Adjust `load_data.py`** if your archive format differs (e.g., parquet
   files, tar.gz, raw directories).
3. **Adjust `preprocess.py`** for your feature engineering needs.
4. **Replace `model.py`** with your model architecture.
5. **Update `config.yaml`** with your bucket, prefix, and hyperparameters.
6. **Scale the PVC** — set storage to match your dataset size + headroom.

The key insight remains: the PVC acts as a persistent, version-aware cache. Your
training pods read from fast local storage, and ZenML handles the orchestration,
credential injection, and metadata tracking.
