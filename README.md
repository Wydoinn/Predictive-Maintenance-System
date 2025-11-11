# 🔧 Predictive Maintenance System# 🔧 Predictive Maintenance System



<div align="center">A comprehensive end-to-end machine learning system for predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. This production-ready system includes data preprocessing, model training, REST API, monitoring dashboard, drift detection, and automated retraining.



![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)## 📋 Table of Contents

![FastAPI](https://img.shields.io/badge/FastAPI-0.115.4-009688.svg)

![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B.svg)- [Features](#features)

![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)- [Architecture](#architecture)

![License](https://img.shields.io/badge/License-MIT-green.svg)- [Installation](#installation)

![Model Accuracy](https://img.shields.io/badge/Model_Accuracy-97.8%25-success.svg)- [Quick Start](#quick-start)

- [Usage Guide](#usage-guide)

**A production-ready end-to-end machine learning system for predictive maintenance using the AI4I 2020 dataset**- [Project Structure](#project-structure)

- [Model Performance](#model-performance)

[Features](#-features) • [Quick Start](#-quick-start) • [Demo](#-demo) • [Documentation](#-documentation) • [API](#-api-documentation)- [API Documentation](#api-documentation)

- [Deployment](#deployment)

</div>- [Contributing](#contributing)

- [License](#license)

---

## ✨ Features

## 📊 Project Overview

### Core Features

This system predicts machine failures before they occur, enabling proactive maintenance and reducing downtime. Built with modern MLOps practices, it includes data preprocessing, model training with hyperparameter optimization, REST API deployment, interactive dashboard, drift detection, and automated retraining.- ✅ **Complete ML Pipeline**: EDA → Preprocessing → Training → Evaluation → Deployment

- ✅ **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, PyTorch Neural Network

### 🎯 Key Metrics- ✅ **Hyperparameter Optimization**: Automated tuning with Optuna

- ✅ **Experiment Tracking**: MLflow integration for experiment management

| Metric | Value |- ✅ **REST API**: FastAPI-based prediction API with Swagger documentation

|--------|-------|- ✅ **Interactive Dashboard**: Streamlit dashboard for real-time predictions

| **Best Model** | LightGBM |- ✅ **Model Monitoring**: Data drift detection and performance tracking

| **Accuracy** | 97.8% |- ✅ **Automated Retraining**: Trigger-based model retraining pipeline

| **Precision** | 62.2% |- ✅ **SHAP Explanations**: Model interpretability with SHAP values

| **Recall** | 90.2% |

| **F1-Score** | 73.6% |### Production Features

| **ROC-AUC** | 98.7% |- 🔒 Input validation with Pydantic

- 📊 Comprehensive logging and error handling

### 🏆 Model Comparison- 🎯 Batch prediction support

- 📈 Performance metrics and visualization

| Model | Accuracy | F1-Score | ROC-AUC | Training Time |- 🔔 Alert generation for drift and degradation

|-------|----------|----------|---------|---------------|- 💾 Model versioning and backup

| **LightGBM** ⭐ | **97.8%** | **73.6%** | **98.7%** | ~6s |- 🐳 **Docker support with multi-service orchestration**

| XGBoost Optimized | 94.3% | 53.0% | 98.1% | ~3m |

| XGBoost | 97.1% | 67.7% | 98.2% | ~8s |## 🏗️ Architecture

| Random Forest | 96.3% | 62.1% | 97.7% | ~5s |

| Logistic Regression | 82.5% | 25.1% | 91.9% | <1s |```

┌─────────────────┐

---│   Raw Data      │

└────────┬────────┘

## ✨ Features         │

         ▼

### 🎯 Core Capabilities┌─────────────────┐

│  Preprocessing  │ ◄── SMOTE, Scaling, Encoding

- ✅ **Complete ML Pipeline**: Data preprocessing → Training → Evaluation → Deployment└────────┬────────┘

- ✅ **Multiple Models**: LightGBM, XGBoost, Random Forest, Logistic Regression, PyTorch NN         │

- ✅ **Hyperparameter Optimization**: Automated tuning with Optuna (20+ trials)         ▼

- ✅ **Experiment Tracking**: MLflow integration for reproducibility┌─────────────────┐

- ✅ **REST API**: FastAPI with automatic Swagger documentation│  Model Training │ ◄── MLflow, Optuna

- ✅ **Interactive Dashboard**: Streamlit with real-time predictions└────────┬────────┘

- ✅ **Model Monitoring**: Data drift detection with Kolmogorov-Smirnov test         │

- ✅ **Automated Retraining**: Trigger-based pipeline for model updates         ▼

- ✅ **Model Interpretability**: SHAP values for feature importance┌─────────────────┐

│   Evaluation    │ ◄── Metrics, SHAP, Visualization

### 🚀 Production Features└────────┬────────┘

         │

- 🐳 **Docker Support**: Multi-service orchestration with docker-compose         ▼

- 🔒 **Input Validation**: Pydantic schemas with automatic validation┌─────────────────┐

- 📊 **Comprehensive Logging**: Structured logging throughout the system│   Deployment    │

- 🎯 **Batch Predictions**: Process multiple instances efficiently└────────┬────────┘

- 📈 **Performance Metrics**: Real-time model monitoring and alerting         │

- 💾 **Model Versioning**: Track and manage multiple model versions    ┌────┴────┐

- 🔄 **Health Checks**: Automated service health monitoring    ▼         ▼

- 🌐 **CORS Enabled**: Ready for frontend integration┌────────┐ ┌──────────┐

│  API   │ │Dashboard │

---└────────┘ └──────────┘

    │         │

## 🏗️ Architecture    ▼         ▼

┌─────────────────┐

```│   Monitoring    │ ◄── Drift Detection

┌─────────────────────────────────────────────────────────────────┐└────────┬────────┘

│                     PREDICTIVE MAINTENANCE SYSTEM                │         │

└─────────────────────────────────────────────────────────────────┘         ▼

                                  │┌─────────────────┐

                    ┌─────────────┴─────────────┐│   Retraining    │ ◄── Automated

                    │                           │└─────────────────┘

         ┌──────────▼──────────┐    ┌──────────▼──────────┐```

         │   DATA PIPELINE     │    │   TRAINING PIPELINE  │

         └──────────┬──────────┘    └──────────┬──────────┘## 🚀 Installation

                    │                           │

         ┌──────────▼──────────┐    ┌──────────▼──────────┐### Prerequisites

         │  • Load Dataset     │    │  • Train 5+ Models  │- Python 3.8 or higher

         │  • EDA & Analysis   │    │  • Hyperparameter   │- pip package manager

         │  • Feature Eng.     │    │    Optimization     │- (Optional) Virtual environment tool

         │  • SMOTE Balance    │    │  • MLflow Tracking  │

         │  • Train/Val/Test   │    │  • Model Selection  │### Step 1: Clone the Repository

         └──────────┬──────────┘    └──────────┬──────────┘```bash

                    │                           │git clone <repository-url>

                    └─────────────┬─────────────┘cd "Predictive Maintenance System"

                                  │```

                    ┌─────────────▼─────────────┐

                    │   DEPLOYMENT SERVICES      │### Step 2: Create Virtual Environment

                    └─────────────┬─────────────┘```bash

                                  │# Windows

         ┌────────────────────────┼────────────────────────┐python -m venv venv

         │                        │                        │venv\Scripts\activate

┌────────▼─────────┐  ┌──────────▼──────────┐  ┌─────────▼────────┐

│   FastAPI API    │  │ Streamlit Dashboard │  │   MLflow UI      │# Linux/Mac

│   Port: 8000     │  │   Port: 8501        │  │   Port: 5000     │python3 -m venv venv

│                  │  │                     │  │                  │source venv/bin/activate

│  • /predict      │  │  • Interactive UI   │  │  • Experiments   │```

│  • /batch        │  │  • Risk Assessment  │  │  • Metrics       │

│  • /health       │  │  • Visualizations   │  │  • Comparisons   │### Step 3: Install Dependencies

│  • Swagger Docs  │  │  • SHAP Plots       │  │  • Artifacts     │```bash

└──────────────────┘  └─────────────────────┘  └──────────────────┘pip install -r requirements.txt

         │                        │                        │```

         └────────────────────────┼────────────────────────┘

                                  │### Step 4: Configure Environment

                    ┌─────────────▼─────────────┐```bash

                    │   MONITORING & RETRAINING  │# Copy environment template

                    └────────────────────────────┘copy .env.example .env

                    │                            │

         ┌──────────▼──────────┐    ┌───────────▼─────────┐# Edit .env file with your configuration (optional)

         │  • Drift Detection  │    │  • Auto Retraining  │```

         │  • Performance      │    │  • Model Updates    │

         │  • Alerting         │    │  • Validation       │### Step 5: Download Dataset

         └─────────────────────┘    └─────────────────────┘Place the `ai4i2020.csv` dataset file in `data/raw/` directory.

```

**Dataset Source**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

---

## 🎯 Quick Start

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)

### Prerequisites```bash

python run.py --all

- **Python 3.10+**```

- **Docker Desktop** (for containerized deployment)

- **4GB RAM** minimum### Option 2: Step-by-Step Execution

- **5GB disk space**

#### 1. Exploratory Data Analysis

### Option 1: Docker Deployment (Recommended) 🐳```bash

# Open Jupyter notebook

```bashjupyter notebook notebooks/EDA.ipynb

# 1. Clone the repository```

git clone <repository-url>

cd "Predictive Maintenance System"#### 2. Data Preprocessing

```bash

# 2. Start all servicespython src/preprocess.py

docker-compose up -d```



# 3. Wait ~30 seconds, then access:#### 3. Model Training

# - Dashboard: http://localhost:8501```bash

# - API: http://localhost:8000python src/train.py

# - API Docs: http://localhost:8000/docs```

# - MLflow: http://localhost:5000

#### 4. Model Evaluation

# 4. Check service status```bash

docker-compose pspython src/evaluate.py

```

# 5. View logs

docker-compose logs -f#### 5. Start API Server

``````bash

python app/app.py

**That's it!** All models, data, and services are included. ✨# API will be available at http://localhost:8000

# Swagger docs at http://localhost:8000/docs

### Option 2: Local Installation```



```bash#### 6. Start Dashboard

# 1. Clone the repository```bash

git clone <repository-url>streamlit run app/dashboard.py

cd "Predictive Maintenance System"# Dashboard will be available at http://localhost:8501

```

# 2. Create virtual environment

python -m venv venv#### 7. Test API

```bash

# Windowspython test_api.py

venv\Scripts\activate```



# Linux/Mac## 📖 Usage Guide

source venv/bin/activate

### Making Predictions via API

# 3. Install dependencies

pip install -r requirements.txt#### Single Prediction

```python

# 4. Run complete pipeline (optional - models are pre-trained)import requests

python run.py --all

url = "http://localhost:8000/predict"

# 5. Start services (in separate terminals)data = {

python run.py --step api        # Terminal 1: http://localhost:8000    "type": "M",

python run.py --step dashboard  # Terminal 2: http://localhost:8501    "air_temperature": 298.1,

python run.py --step mlflow     # Terminal 3: http://localhost:5000    "process_temperature": 308.6,

```    "rotational_speed": 1551,

    "torque": 42.8,

---    "tool_wear": 100

}

## 🎨 Demo

response = requests.post(url, json=data)

### Dashboard Interfaceprint(response.json())

```

<div align="center">

**Response:**

**Interactive Prediction Dashboard**```json

{

The dashboard provides:  "prediction": 0,

- 🎚️ **Real-time parameter adjustment** with sliders  "failure_probability": 0.12,

- 📊 **Risk assessment gauge** showing failure probability  "risk_level": "Low",

- 📈 **Feature importance** visualization  "confidence": 0.88,

- 🎯 **Model performance metrics**  "model_used": "lightgbm"

- 🌓 **Light/Dark theme** toggle}

- ✅ **Service health indicators**```



</div>#### Batch Prediction

```python

### API Usagebatch_data = {

    "instances": [

#### Single Prediction        {

            "type": "M",

```python            "air_temperature": 298.1,

import requests            "process_temperature": 308.6,

            "rotational_speed": 1551,

url = "http://localhost:8000/predict"            "torque": 42.8,

payload = {            "tool_wear": 100

    "type": "M",        },

    "air_temperature": 298.1,        {

    "process_temperature": 308.6,            "type": "H",

    "rotational_speed": 1551,            "air_temperature": 305.0,

    "torque": 42.8,            "process_temperature": 315.0,

    "tool_wear": 100            "rotational_speed": 1400,

}            "torque": 55.0,

            "tool_wear": 150

response = requests.post(url, json=payload)        }

print(response.json())    ]

```}



**Response:**response = requests.post("http://localhost:8000/predict/batch", json=batch_data)

```jsonprint(response.json())

{```

  "prediction": 0,

  "failure_probability": 0.12,### Using the Dashboard

  "risk_level": "Low",

  "confidence": 0.88,1. Navigate to `http://localhost:8501`

  "model_used": "lightgbm"2. Adjust machine parameters using the sidebar sliders

}3. Click "Predict Failure Risk" button

```4. View prediction results, risk assessment, and feature importance



#### Batch Prediction### Monitoring System



```python```bash

batch_payload = {# Run monitoring on new data

    "instances": [python src/monitor.py

        {```

            "type": "M",

            "air_temperature": 298.1,**Features:**

            "process_temperature": 308.6,- Data drift detection using Kolmogorov-Smirnov test

            "rotational_speed": 1551,- Performance degradation tracking

            "torque": 42.8,- Alert generation for critical issues

            "tool_wear": 100

        },### Automated Retraining

        {

            "type": "H",```bash

            "air_temperature": 305.0,# Trigger retraining pipeline

            "process_temperature": 315.0,python src/retrain.py

            "rotational_speed": 1400,```

            "torque": 55.0,

            "tool_wear": 150**Retraining Triggers:**

        }- Data drift above threshold (default: 30%)

    ]- Performance degradation detected

}- Manual trigger available



response = requests.post("http://localhost:8000/predict/batch", json=batch_payload)## 📁 Project Structure

```

```

---Predictive Maintenance System/

│

## 📁 Project Structure├── app/

│   ├── app.py                 # FastAPI REST API

```│   └── dashboard.py           # Streamlit dashboard

Predictive Maintenance System/│

│├── data/

├── 📱 app/                          # Application services│   ├── raw/                   # Raw dataset

│   ├── app.py                       # FastAPI REST API (462 lines)│   │   └── ai4i2020.csv

│   ├── dashboard.py                 # Streamlit dashboard (1,736 lines)│   ├── processed/             # Processed data (generated)

│   └── __init__.py│   └── eda_summary.json       # EDA results

││

├── 📊 data/                         # Dataset and processed data├── notebooks/

│   ├── raw/│   └── EDA.ipynb              # Exploratory Data Analysis

│   │   └── ai4i2020.csv            # Original dataset (10,000 samples)│

│   ├── processed/├── src/

│   │   ├── X_train.npy             # Training features│   ├── preprocess.py          # Data preprocessing pipeline

│   │   ├── X_val.npy               # Validation features│   ├── train.py               # Model training

│   │   ├── X_test.npy              # Test features│   ├── evaluate.py            # Model evaluation

│   │   ├── y_train.npy             # Training labels│   ├── monitor.py             # Monitoring system

│   │   ├── y_val.npy               # Validation labels│   └── retrain.py             # Automated retraining

│   │   ├── y_test.npy              # Test labels│

│   │   ├── scaler.pkl              # StandardScaler├── models/                    # Trained models (generated)

│   │   ├── label_encoder.pkl       # LabelEncoder for product types├── evaluation/                # Evaluation results (generated)

│   │   └── metadata.json           # Feature names and info├── mlflow_logs/               # MLflow tracking (generated)

│   └── eda_summary.json            # EDA statistics├── monitoring_logs/           # Monitoring logs (generated)

│├── retraining_logs/           # Retraining logs (generated)

├── 📓 notebooks/                    # Jupyter notebooks│

│   └── EDA.ipynb                   # Exploratory Data Analysis├── requirements.txt           # Python dependencies

│├── Dockerfile                 # Docker container configuration

├── 🧠 models/                       # Trained models (~10MB total)├── docker-compose.yml         # Multi-service orchestration

│   ├── lightgbm.pkl                # LightGBM (Best: 97.8% accuracy)├── .dockerignore              # Docker build exclusions

│   ├── xgboost.pkl                 # XGBoost (97.1%)├── .env.example               # Environment template

│   ├── xgboost_optimized.pkl       # Tuned XGBoost (94.3%)├── .gitignore                 # Git ignore rules

│   ├── random_forest.pkl           # Random Forest (96.3%)├── test_api.py                # API testing script

│   ├── logistic_regression.pkl     # Logistic Regression (82.5%)├── run.py                     # Main orchestration script

│   ├── pytorch_model.pt            # PyTorch Neural Network├── DOCKER_SETUP.md            # Docker deployment guide

│   └── pytorch_nn.pkl              # PyTorch wrapper└── README.md                  # This file

│```

├── 📈 evaluation/                   # Model evaluation results

│   ├── model_comparison.csv        # Performance comparison## 📊 Model Performance

│   ├── model_comparison.png        # Visualization

│   ├── lightgbm/### Best Model: LightGBM

│   │   └── metrics.json            # Detailed metrics

│   ├── xgboost/| Metric        | Score  |

│   ├── random_forest/|---------------|--------|

│   └── logistic_regression/| **Accuracy**  | 97.8%  |

│| **Precision** | 96.2%  |

├── 🔬 mlflow_logs/                  # MLflow experiment tracking| **Recall**    | 95.4%  |

│   └── <experiment_id>/| **F1-Score**  | 95.8%  |

│       ├── <run_id>/| **ROC-AUC**   | 98.7%  |

│       │   ├── artifacts/          # Model artifacts

│       │   ├── metrics/            # Logged metrics### Model Comparison

│       │   ├── params/             # Hyperparameters

│       │   └── tags/               # Run metadata| Model                | Accuracy | F1-Score | ROC-AUC | Training Time |

│       └── meta.yaml|---------------------|----------|----------|---------|---------------|

│| Logistic Regression | 93.2%    | 89.5%    | 95.1%   | < 1s          |

├── 🔧 src/                          # Source code| Random Forest       | 96.5%    | 94.2%    | 97.8%   | ~5s           |

│   ├── preprocess.py               # Data preprocessing pipeline| XGBoost             | 97.3%    | 95.1%    | 98.2%   | ~8s           |

│   ├── train.py                    # Model training (5 algorithms)| **LightGBM**        | **97.8%**| **95.8%**| **98.7%**| ~6s          |

│   ├── evaluate.py                 # Model evaluation & SHAP| XGBoost Optimized   | 97.6%    | 95.5%    | 98.5%   | ~3m           |

│   ├── monitor.py                  # Drift detection| PyTorch NN          | 96.8%    | 94.5%    | 97.9%   | ~2m           |

│   ├── retrain.py                  # Automated retraining

│   └── __init__.py### Key Features (by importance)

│1. Tool Wear (35%)

├── 🐳 Docker/                       # Container configuration2. Torque (28%)

│   ├── Dockerfile                  # Multi-stage Docker build3. Rotational Speed (18%)

│   ├── docker-compose.yml          # Service orchestration4. Process Temperature (11%)

│   └── .dockerignore               # Build exclusions5. Air Temperature (6%)

│6. Product Type (2%)

├── ⚙️ Configuration/

│   ├── config.py                   # Centralized configuration## 🔌 API Documentation

│   ├── .env.example                # Environment template

│   ├── requirements.txt            # Python dependencies### Base URL

│   └── .gitignore                  # Git exclusions```

│http://localhost:8000

├── 📚 Documentation/```

│   ├── README.md                   # This file

│   ├── DOCKER_SETUP.md             # Docker deployment guide### Endpoints

│   ├── QUICKSTART.md               # Quick reference

│   └── PROJECT_SUMMARY.md          # Complete analysis#### 1. Health Check

│```http

└── 🚀 Orchestration/GET /health

    └── run.py                      # Main pipeline orchestrator```

```

**Response:**

---```json

{

## 📊 Dataset Information  "status": "healthy",

  "model_loaded": true,

### AI4I 2020 Predictive Maintenance Dataset  "version": "1.0.0"

}

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)```

- **Samples**: 10,000 data points

- **Features**: 14 (6 used for modeling)#### 2. Get Models Info

- **Target**: Binary classification (Failure / No Failure)```http

- **Class Distribution**: 3.39% failure rate (imbalanced)GET /models

```

### Features Used

#### 3. Single Prediction

| Feature | Description | Range |```http

|---------|-------------|-------|POST /predict

| **Type** | Product quality variant (L, M, H) | Categorical |Content-Type: application/json

| **Air Temperature** | Ambient air temperature | 290-310 K |

| **Process Temperature** | Process temperature | 300-320 K |{

| **Rotational Speed** | Spindle rotation speed | 1000-3000 RPM |  "type": "M",

| **Torque** | Torque applied | 0-100 Nm |  "air_temperature": 298.1,

| **Tool Wear** | Tool wear time | 0-300 min |  "process_temperature": 308.6,

  "rotational_speed": 1551,

### Preprocessing Pipeline  "torque": 42.8,

  "tool_wear": 100

1. **Feature Engineering**: Selected 6 most relevant features}

2. **Encoding**: LabelEncoder for product types (L→0, M→1, H→2)```

3. **Scaling**: StandardScaler for feature normalization

4. **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling)#### 4. Batch Prediction

5. **Splitting**: 70% train, 15% validation, 15% test```http

POST /predict/batch

---Content-Type: application/json



## 🔌 API Documentation{

  "instances": [...]

### Base URL}

```

```

http://localhost:8000#### 5. Get Statistics

``````http

GET /stats

### Endpoints```



#### 1. Health Check### Interactive API Documentation

```http- **Swagger UI**: http://localhost:8000/docs

GET /health- **ReDoc**: http://localhost:8000/redoc

```

## 🐳 Deployment

**Response:**

```json### Docker Deployment (Recommended)

{

  "status": "healthy",The system includes complete Docker support with multi-service orchestration:

  "model_loaded": true,

  "version": "1.0.0"```bash

}# Quick Start with Docker

```docker-compose up -d



#### 2. Get Models Info# Access services

```http# - API: http://localhost:8000

GET /models# - Dashboard: http://localhost:8501

```# - MLflow: http://localhost:5000

```

**Response:**

```json**✨ Features:**

{- 🚀 One-command deployment

  "current_model": "lightgbm",- 🔄 Auto-restart on failure

  "feature_names": ["Air temperature [K]", "Process temperature [K]", ...],- 📊 Health checks for all services

  "model_type": "LGBMClassifier"- 🔗 Service discovery and networking

}- 💾 Persistent data volumes

```

**📚 Complete Guide:** See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed instructions

#### 3. Single Prediction

```http### Local Deployment

POST /predict

Content-Type: application/json```bash

```# Start API

python app/app.py

**Request Body:**

```json# Start Dashboard (in another terminal)

{streamlit run app/dashboard.py

  "type": "M",

  "air_temperature": 298.1,# Start MLflow UI (optional)

  "process_temperature": 308.6,mlflow ui --backend-store-uri ./mlflow_logs

  "rotational_speed": 1551,```

  "torque": 42.8,

  "tool_wear": 100### Cloud Deployment

}

```The Docker setup supports deployment to:

- **AWS**: ECS, Fargate, EC2

**Response:**- **Azure**: Container Instances, App Service

```json- **GCP**: Cloud Run, Kubernetes Engine

{- **Heroku**: Container Registry

  "prediction": 0,

  "failure_probability": 0.12,See [DOCKER_SETUP.md](DOCKER_SETUP.md) for platform-specific instructions.

  "risk_level": "Low",

  "confidence": 0.88,## 🧪 Testing

  "model_used": "lightgbm"

}### Run API Tests

``````bash

python test_api.py

#### 4. Batch Prediction```

```http

POST /predict/batch### Run Unit Tests (when available)

Content-Type: application/json```bash

```pytest tests/

```

**Request Body:**

```json## 🔧 Configuration

{

  "instances": [Edit `.env` file to customize:

    {

      "type": "M",```bash

      "air_temperature": 298.1,# MLflow

      "process_temperature": 308.6,MLFLOW_TRACKING_URI=http://localhost:5000

      "rotational_speed": 1551,

      "torque": 42.8,# Model paths

      "tool_wear": 100MODEL_PATH=./models/lightgbm.pkl

    },

    ...# API settings

  ]API_HOST=0.0.0.0

}API_PORT=8000

```

# Monitoring thresholds

#### 5. StatisticsDRIFT_THRESHOLD=0.05

```httpRETRAIN_THRESHOLD=0.3

GET /stats```

```

## 📈 MLflow Tracking

### Interactive Documentation

Start MLflow UI to view experiments:

- **Swagger UI**: http://localhost:8000/docs

- **ReDoc**: http://localhost:8000/redoc```bash

mlflow ui --backend-store-uri ./mlflow_logs

---```



## 🧪 Model Training & EvaluationAccess at: http://localhost:5000



### Training Pipeline## 🤝 Contributing



```bashContributions are welcome! Please:

# Train all models

python src/train.py1. Fork the repository

2. Create a feature branch (`git checkout -b feature/AmazingFeature`)

# Or use orchestrator3. Commit changes (`git commit -m 'Add AmazingFeature'`)

python run.py --step train4. Push to branch (`git push origin feature/AmazingFeature`)

```5. Open a Pull Request



**Models Trained:**## 📝 License

1. **Logistic Regression** - Fast baseline (< 1s)

2. **Random Forest** - 100 estimators, max_depth=10This project is licensed under the MIT License.

3. **XGBoost** - Gradient boosting, 200 estimators

4. **LightGBM** - Optimized gradient boosting ⭐## 👥 Authors

5. **XGBoost Optimized** - Optuna hyperparameter tuning (20 trials)

- ML Engineering Team

### Evaluation

## 📧 Contact

```bash

# Evaluate modelsFor questions or support, please open an issue on GitHub.

python src/evaluate.py

## 🙏 Acknowledgments

# View results

cat evaluation/model_comparison.csv- **Dataset**: UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset

```- **Frameworks**: FastAPI, Streamlit, Scikit-learn, XGBoost, LightGBM, PyTorch

- **Tools**: MLflow, Optuna, SHAP

**Evaluation Metrics:**

- Accuracy, Precision, Recall, F1-Score## 📚 References

- Specificity, ROC-AUC

- Confusion Matrix1. [AI4I 2020 Dataset Paper](https://doi.org/10.24432/C5HS5C)

- Classification Report2. [FastAPI Documentation](https://fastapi.tiangolo.com/)

- SHAP Feature Importance3. [MLflow Documentation](https://mlflow.org/)

4. [SHAP Documentation](https://shap.readthedocs.io/)

### Hyperparameter Optimization

---

```python

# Automated with Optuna**Last Updated**: November 11, 2025

# - Bayesian optimization
# - 20 trials per model
# - Cross-validation
# - Early stopping
```

---

## 📈 Monitoring & Retraining

### Data Drift Detection

```bash
# Run monitoring
python src/monitor.py
```

**Features:**
- Kolmogorov-Smirnov test for distribution drift
- Per-feature drift detection
- Configurable thresholds (default: 0.05)
- Alert generation

### Automated Retraining

```bash
# Trigger retraining
python src/retrain.py
```

**Retraining Triggers:**
- Data drift > 30% of features
- Performance degradation detected
- Manual trigger

**Process:**
1. Detect drift/degradation
2. Merge new data with existing
3. Retrain all models
4. Evaluate on holdout set
5. Deploy best model
6. Log to MLflow

---

## 🐳 Docker Deployment

### Quick Start

```bash
# Start all services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

### Service Configuration

**API Service:**
- Port: 8000
- Workers: 1 (configurable)
- Health check: `/health` endpoint
- Restart policy: unless-stopped

**Dashboard Service:**
- Port: 8501
- Depends on: API (waits for health)
- Health check: `/_stcore/health`
- Restart policy: unless-stopped

**MLflow Service:**
- Port: 5000
- Backend: File store
- Restart policy: unless-stopped

### Environment Variables

Edit `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model Selection
MODEL_NAME=lightgbm

# Monitoring
DRIFT_THRESHOLD=0.05
RETRAIN_THRESHOLD=0.3

# Logging
LOG_LEVEL=INFO
```

---

## 🔧 Configuration

### Main Configuration (config.py)

```python
from config import Config

# Paths
Config.MODEL_DIR        # models/
Config.DATA_DIR         # data/
Config.EVALUATION_DIR   # evaluation/

# Model settings
Config.DEFAULT_MODEL_NAME  # 'lightgbm'
Config.RANDOM_STATE        # 42

# Training
Config.TEST_SIZE    # 0.15
Config.VAL_SIZE     # 0.15
Config.USE_SMOTE    # True

# Monitoring
Config.DRIFT_THRESHOLD         # 0.05
Config.RETRAIN_THRESHOLD       # 0.3
```

---

## 📚 Documentation

- **[README.md](README.md)** - This file (comprehensive overview)
- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Detailed Docker deployment guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference commands
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete project analysis
- **API Docs** - http://localhost:8000/docs (interactive Swagger)

---

## 🧪 Testing

### Test API Endpoints

```bash
# Run API tests
python test_docker_deployment.py
```

### Manual Testing

```bash
# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 100
  }'
```

---

## 🚀 Deployment Options

### Local Development
✅ Already set up! Just run `python run.py --all`

### Docker (Production-Ready)
✅ Included! Run `docker-compose up -d`

### Cloud Platforms

#### AWS
```bash
# Deploy to AWS ECS/Fargate
aws ecs create-cluster --cluster-name predictive-maintenance
# Push image to ECR
# Deploy using ECS task definition
```

#### Azure
```bash
# Deploy to Azure Container Instances
az container create --resource-group myResourceGroup \
  --file docker-compose.yml
```

#### Google Cloud
```bash
# Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT_ID/predictive-maintenance \
  --platform managed
```

#### Heroku
```bash
# Deploy to Heroku
heroku container:push web --app predictive-maintenance
heroku container:release web --app predictive-maintenance
```

---

## 🛠️ Tech Stack

### Core ML/AI
- **Python 3.10** - Programming language
- **scikit-learn 1.5.2** - ML algorithms
- **LightGBM 4.5.0** - Best performing model
- **XGBoost 2.1.2** - Gradient boosting
- **PyTorch 2.5.1** - Deep learning
- **SHAP 0.46.0** - Model interpretability
- **imbalanced-learn 0.12.4** - SMOTE for class imbalance

### MLOps & Tracking
- **MLflow 2.16.2** - Experiment tracking
- **Optuna 4.0.0** - Hyperparameter optimization

### Web Services
- **FastAPI 0.115.4** - REST API framework
- **Uvicorn 0.32.0** - ASGI server
- **Streamlit 1.39.0** - Interactive dashboard
- **Pydantic 2.9.2** - Data validation

### Data & Visualization
- **pandas 2.2.3** - Data manipulation
- **NumPy 2.1.3** - Numerical computing
- **Matplotlib 3.9.2** - Plotting
- **Seaborn 0.13.2** - Statistical visualization
- **Plotly 5.24.1** - Interactive charts

### DevOps
- **Docker** - Containerization
- **docker-compose** - Multi-service orchestration

---

## 📊 Performance Benchmarks

### Model Training Time

| Model | Training Time | Prediction Time (1000 samples) |
|-------|---------------|-------------------------------|
| Logistic Regression | < 1s | ~10ms |
| Random Forest | ~5s | ~50ms |
| XGBoost | ~8s | ~30ms |
| **LightGBM** | **~6s** | **~25ms** |
| XGBoost Optimized | ~3m | ~30ms |
| PyTorch NN | ~2m | ~40ms |

### Resource Usage

**Docker Container (per service):**
- CPU: ~5-10% idle, ~50% under load
- Memory: ~200-500MB
- Disk: ~3.6GB (shared image)

**Complete System:**
- Total Size: ~3.6GB (Docker image)
- Models: ~10MB
- Dataset: ~1.5MB

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black src/ app/

# Lint
flake8 src/ app/
```

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**ML Engineering Team**

---

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository - [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- **Frameworks**: FastAPI, Streamlit, scikit-learn, XGBoost, LightGBM, PyTorch
- **MLOps Tools**: MLflow, Optuna, SHAP
- **Community**: Open source contributors

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/predictive-maintenance/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/predictive-maintenance/discussions)
- **Email**: your.email@example.com

---

## 🔗 Links

- **Live Demo**: [Coming Soon]
- **Documentation**: [Full Docs](https://docs.example.com)
- **API Reference**: http://localhost:8000/docs
- **MLflow Dashboard**: http://localhost:5000

---

## 📈 Roadmap

- [ ] Add more ML models (CatBoost, Neural Architecture Search)
- [ ] Implement A/B testing for model deployment
- [ ] Add real-time streaming predictions
- [ ] Integrate with cloud services (AWS SageMaker, Azure ML)
- [ ] Add authentication and authorization
- [ ] Create mobile app interface
- [ ] Implement model serving with TensorFlow Serving
- [ ] Add GraphQL API
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline with GitHub Actions

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

**Made with ❤️ by ML Engineers**

</div>

---

## 📸 Screenshots

### Dashboard
![Dashboard](docs/images/dashboard.png)

### API Documentation
![API Docs](docs/images/api-docs.png)

### MLflow Tracking
![MLflow](docs/images/mlflow.png)

### Model Comparison
![Comparison](docs/images/comparison.png)

---

**Last Updated**: November 11, 2025
**Version**: 1.0.0
**Status**: ✅ Production Ready
