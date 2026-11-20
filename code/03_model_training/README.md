# Model Training

This directory contains the minimal setup for training and testing the feasibility classifiers of **BRMapper**. The classifiers decide which of the atom-map candidates (ITS graphs) are chemically feasible. Two model families are supported:

- **Graph-kernel SVMs** — SVMs with graph kernels WL-OA, NSPD, and NH
- **GNNs** — GIN, GAT, and GraphSAGE

## Project Structure

```text
03_model_training/
├── train_test_svm.py        # Train and test graph-kernel SVMs
├── train_test_gnn.py        # Train and test GNNs
├── src/
│   ├── utils.py             # Dataset loading, subsetting, undersampling
│   ├── svm.py               # GraKeL dataset conversion, kernels, SVM evaluation
│   ├── models.py            # GIN, GAT, GraphSAGE models
│   ├── metric.py            # Classification metrics
│   └── mlflow_helper.py     # MLflow experiment setup and dataset logging
├── cfg/
│   ├── train_test_svm.yaml  # SVM run defaults
│   ├── train_test_gnn.yaml  # GNN run defaults
│   ├── model/               # Kernel/model configurations
│   ├── ds_train/            # Training dataset configs
│   └── ds_test/             # Test dataset configs
```

## Data

The training and test datasets are read directly from the preprocessing pipeline in `code/02_data_preprocessing`. After running that pipeline with the default output directory, its TU-format datasets are located in `../02_data_preprocessing/results/TUData/`, which is where the default dataset configs (`cfg/ds_train/tudata.yaml`, `cfg/ds_test/tudata.yaml`) point (`train_TUdata` and `test_TUdata`). 

## Usage

Build the unified container image (tagged `biermapper:latest`) from the **repository root**. It contains the dependencies for all three pipeline stages:

```bash
docker build -t biermapper:latest ..
```

Training runs are configured via [Hydra](https://hydra.cc/). The dataset, model, and run parameters are selected via the command line. All commands below are executed from the `code/03_model_training` directory; for GPU support add `--gpus all` to the `docker run` call.

### SVMs

```bash
docker run --rm --user $UID:$GID -v ..:/workspace -w /workspace/code/03_model_training biermapper:latest python train_test_svm.py +ds_train=tudata +ds_test=tudata +model=NSPD undersampling_ratio=0.1
```

If all dependencies are installed locally (see the root `requirements.txt`), the same runs work with plain `python train_test_svm.py ...`.

Available graph kernels: `+model=WL-OA`, `+model=NSPD`, `+model=NH`. Kernel and SVM hyperparameters are defined in `cfg/model/` and can be overridden with their full config path, e.g. `+model=WL-OA model.kernel.n_iter=5 model.svm.C=10`.

Relevant parameters (defaults from `cfg/train_test_svm.yaml`):

| Parameter | Description | Default | Example |
|---|---|---|---|
| `runs` | Number of repeated train/test runs | `20` | `runs=1` |
| `scale_C` | Scale the SVM C value by 1/sqrt(n_train) | `True` | `scale_C=False` |
| `subset` | Size of a random stratified subset of the training data | `null` | `subset=5000` |
| `undersampling_ratio` | Ratio of positive (feasible) samples in the training data after undersampling | `null` | `undersampling_ratio=0.1` |

`subset` and `undersampling_ratio` are mutually exclusive; by default both are `null` (no resizing of the training data).

### GNNs

```bash
docker run --rm --user $UID:$GID -v ..:/workspace -w /workspace/code/03_model_training biermapper:latest python train_test_gnn.py +ds_train=tudata +ds_test=tudata +model=GIN
```

Available models: `+model=GIN`, `+model=GAT`, `+model=GraphSAGE`. The model config (`cfg/model/`) defines the architecture as well as the training parameters (learning rate, epochs, batch size). They can be overridden on the command line:

| Parameter | Description | Default | Example |
|---|---|---|---|
| `runs` | Number of repeated train/test runs | `20` | `runs=1` |
| `model.lr` | Learning rate (Adam) | `0.001` | `model.lr=0.0045` |
| `model.epochs` | Training epochs per run | `100` | `model.epochs=50` |
| `model.batch_size` | Batch size | `1024` | `model.batch_size=512` |
| `undersampling_ratio` | Ratio of positive (feasible) samples in the training data after undersampling | `null` | `undersampling_ratio=0.1` |

Model architecture parameters (layers, sizes, pooling, etc.) are overridden the same way, e.g. `+model=GAT model.num_layers=3 model.dropout=0.1`.

## Experiment Tracking

All runs, metrics, and prediction artifacts are logged to [MLflow](https://mlflow.org/). By default, a local SQLite database (`./mlflow.db`, with artifacts in `./mlartifacts/`) is used. To log to a tracking server, set the `MLFLOW_TRACKING_URI` environment variable accordingly:

```bash
docker run --rm --user $UID:$GID -e MLFLOW_TRACKING_URI=http://my-tracking-server:5000 -v ..:/workspace -w /workspace/code/03_model_training biermapper:latest python train_test_svm.py ...
```
