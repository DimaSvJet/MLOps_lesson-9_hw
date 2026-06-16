# MLOps Homework 09 – MLflow, ArgoCD, Prometheus & Grafana on EKS

## Overview

This project demonstrates a complete MLOps workflow deployed on AWS EKS using GitOps principles with ArgoCD.

The solution includes:

- Kubernetes cluster (AWS EKS)
- ArgoCD for GitOps deployment
- PostgreSQL as MLflow backend store
- MinIO as S3-compatible artifact storage
- MLflow Tracking Server
- Prometheus PushGateway for custom metrics
- Prometheus for metrics collection
- Grafana for visualization

The project trains a machine learning model on the Iris dataset, logs experiments to MLflow, stores artifacts in MinIO, and exports model metrics to Prometheus/Grafana.

---

## Architecture

```text
GitHub Repository
        |
        v
     ArgoCD
        |
        +---------------------+
        |                     |
        v                     v
     MinIO              PostgreSQL
        \                   /
         \                 /
          \               /
            MLflow Server
                   |
                   v
         train_and_push.py
                   |
                   v
             PushGateway
                   |
                   v
              Prometheus
                   |
                   v
                Grafana
```

---

## Components

### ArgoCD

ArgoCD continuously synchronizes Kubernetes resources from the Git repository and deploys applications automatically.

Applications deployed:

- MinIO
- PostgreSQL
- MLflow
- PushGateway

---

### MinIO

MinIO is used as an S3-compatible object storage.

Responsibilities:

- Store MLflow artifacts
- Store trained model files
- Store experiment outputs

Bucket:

```text
mlflow-artifacts
```

---

### PostgreSQL

PostgreSQL serves as the MLflow backend database.

Stores:

- Experiments
- Runs
- Parameters
- Metrics
- Metadata

---

### MLflow

MLflow Tracking Server is used to:

- Track experiments
- Store parameters
- Store metrics
- Register artifacts

Tracked metrics:

- accuracy
- loss

Logged artifacts:

- Trained model (.joblib)

---

### Prometheus PushGateway

Custom experiment metrics are pushed from the training script to PushGateway.

Exported metrics:

```text
mlflow_accuracy
mlflow_loss
```

---

### Prometheus

Prometheus scrapes metrics from PushGateway and stores them for monitoring and visualization.

---

### Grafana

Grafana visualizes experiment metrics collected by Prometheus.

Implemented dashboards:

- MLflow Accuracy
- MLflow Loss

---

## Machine Learning Experiment

Dataset:

```text
Iris Dataset
```

Model:

```python
MLPClassifier
```

Tracked parameters:

- learning_rate
- epochs

Tracked metrics:

- accuracy
- loss

Best model is automatically selected and saved.

Output:

```text
best_model/model.joblib
```

---

## Running the Experiment

Activate virtual environment:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r mlops-experiments/argocd/experiments/requirements.txt
```

Run training:

```bash
python mlops-experiments/argocd/experiments/train_and_push.py
```

---

## Monitoring

Prometheus metrics:

```text
mlflow_accuracy
mlflow_loss
```

Grafana Explore examples:

```promql
mlflow_accuracy
```

```promql
mlflow_loss
```

---

## Results

The implemented solution demonstrates:

- GitOps deployment using ArgoCD
- Experiment tracking with MLflow
- Artifact storage using MinIO
- Metadata storage using PostgreSQL
- Metrics export through PushGateway
- Monitoring using Prometheus
- Visualization using Grafana

The complete MLOps workflow is deployed and operational on AWS EKS.
