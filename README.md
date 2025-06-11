# Retail AI Suite

A modular AI project focused on solving key retail business problems using machine learning.

---

## Module 1: Demand Forecasting

This module builds a supervised ML model using XGBoost to predict future product sales based on historical data — a crucial tool for retail inventory planning and management.

### Features
- Data preprocessing and feature engineering
- Train/test split and model training using XGBoost
- Evaluation with RMSE, MAE, and R² metrics
- Visualization of actual vs predicted sales
- Save and reuse trained model
- REST API endpoint for real-time sales prediction (FastAPI)

---

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment tool (venv recommended)

### Installation

```bash
git clone https://github.com/mnalluri19/retail-ai-suite.git
cd retail-ai-suite
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
