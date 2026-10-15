# TriageIQ
Intelligent complaint classification and prioritization for 311 service requests.

## Pipeline
1. EDA (notebooks/01_eda.ipynb)
2. Baseline TF-IDF + Logistic Regression (notebooks/02_baseline.ipynb)
3. DistilBERT comparison (planned)
4. FastAPI + Docker + MLflow (planned)
5. Drift monitoring (planned)

## Setup
python -m venv .venv
pip install -r requirements.txt

## Data
Synthetic 311 dataset (20,000 rows). Place it at data/311_requests.csv.
