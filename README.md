# 🔧 Predictive Maintenance System

<div align="center">

**An end-to-end MLOps-driven machine learning system for predicting equipment failures using the AI4I 2020 Predictive Maintenance Dataset.**
Includes automated training, REST API, dashboard, monitoring, drift detection, and retraining — fully containerized with Docker.

</div>


## 🚀 Overview

The **Predictive Maintenance System** forecasts machine failures before they happen — enabling proactive maintenance and reducing downtime.
It follows modern **MLOps best practices** with modular pipelines, model tracking, and production-ready deployment.


## ✨ Features

* 🧠 **End-to-End ML Pipeline** – Data → Training → Evaluation → Deployment
* ⚙️ **Multiple Models** – Logistic Regression, Random Forest, XGBoost, LightGBM, PyTorch NN
* 🎯 **Optuna Tuning + MLflow Tracking** – Automated optimization & experiment management
* 🌐 **FastAPI REST API** – Real-time predictions + Swagger docs
* 📊 **Streamlit Dashboard** – Interactive visualization & failure risk assessment
* 🔍 **Drift Detection + Auto Retraining** – Keeps models accurate over time
* 🧩 **Explainability** – SHAP-based feature importance
* 🐳 **Dockerized Deployment** – Multi-service orchestration with `docker-compose`


## 🧩 Architecture

```
Raw Data → Preprocessing → Model Training → Evaluation
                    ↓
        API  ↔  Dashboard  ↔  Monitoring
                    ↓
            Automated Retraining
```


## 🏆 Model Performance

| Model               | Accuracy   | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- |
| **LightGBM ⭐**      | **97.8 %** | **73.6 %** | **98.7 %** |
| XGBoost             | 97.1 %     | 67.7 %     | 98.2 %     |
| Random Forest       | 96.3 %     | 62.1 %     | 97.7 %     |
| Logistic Regression | 82.5 %     | 25.1 %     | 91.9 %     |


## ⚙️ Quick Start

### 🐳 Run with Docker (Recommended)

```bash
git clone https://github.com/Wydoinn/Predictive-Maintenance-System.git
cd Predictive-Maintenance-System
docker-compose up -d
```

**Access Services**

* API → [http://localhost:8000](http://localhost:8000)
* Docs → [http://localhost:8000/docs](http://localhost:8000/docs)
* Dashboard → [http://localhost:8501](http://localhost:8501)
* MLflow → [http://localhost:5001](http://localhost:5001)

### 💻 Local Setup

```bash
git clone https://github.com/Wydoinn/Predictive-Maintenance-System.git
cd Predictive-Maintenance-System
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

#### 🔧 Usage

```bash
python run.py --all                   # Run complete pipeline
python run.py --step preprocess       # Run preprocessing only
python run.py --step train            # Run training only
python run.py --step evaluate         # Run evaluation only
python run.py --step api              # Start API server
python run.py --step dashboard        # Start dashboard
python run.py --step monitor          # Run monitoring
python run.py --step retrain          # Run retraining
```


## 📊 API Example

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 100
}

print(requests.post(url, json=data).json())
```

**Response**

```json
{
  "prediction": 0,
  "failure_probability": 0.12,
  "risk_level": "Low",
  "model_used": "lightgbm"
}
```

## 🎨 Dashboard Preview

<img width="1897" height="913" alt="Screenshot 2025-11-11 171731" src="https://github.com/user-attachments/assets/cad10b41-e870-4033-adc9-21eae0620349" />

<img width="1908" height="919" alt="Screenshot 2025-11-11 171836" src="https://github.com/user-attachments/assets/0b71ee6f-228b-4727-b427-a289a4fe1ede" />


## 🧠 Tech Stack

- **ML & AI** – scikit-learn | LightGBM | XGBoost | PyTorch | SHAP
- **MLOps** – MLflow | Optuna | Pydantic
- **Web** – FastAPI | Streamlit | Uvicorn
- **DevOps** – Docker | docker-compose


## 🧾 Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
**Samples:** 10 000  **Target:** Failure / No Failure  **Type:** Binary classification


## 📁 Project Structure

```
Predictive-Maintenance-System/
├── app/                # FastAPI & Streamlit apps
├── src/                # Training, evaluation, monitoring, retraining
├── data/               # Raw & processed data
├── models/             # Saved models
├── evaluation/         # Model performance logs
├── mlflow_logs/        # Experiment tracking
├── run.py              # Orchestrator
└── docker-compose.yml  # Deployment config
```


## 📈 Monitoring & Retraining

```bash
python src/monitor.py    # Detect data drift
python src/retrain.py    # Auto-retrain models
```


## 📝 License

Licensed under the **MIT License** — free for personal & commercial use.
