<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## 🚴‍♂️ Project: Ecobici Bike Sharing Demand Forecasting

This repository contains an MLOps project focused on predicting bike demand for Mexico City's Ecobici bike-sharing system using time series forecasting and machine learning techniques.

### 📊 About the Dataset

**Ecobici Public Dataset**
- **Source**: Mexico City Government's open data portal
- **System**: Ecobici bike-sharing program in Mexico City
- **Data Type**: Bike trip records including station locations, pickup times, and ride counts
- **Time Period**: June-August 2025 (configurable)
- **Granularity**: Hourly aggregated ride counts per station




### Set up your environment
In this course, we use Databricks 16.04 LTS runtime, which uses Python 3.12.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

Install task: https://taskfile.dev/installation/

Update .env file with the following:
```
GIT_TOKEN=<your github PAT>
```

To create a new environment and create a lockfile, run:
```
task sync-dev
source .venv/bin/activate
```

Or, alternatively:
```
export GIT_TOKEN=<your github PAT>
uv venv -p 3.11 .venv
source .venv/bin/activate
uv sync --extra dev
```

*This project is part of the Marvelous MLOps course, demonstrating end-to-end MLOps practices with Databricks and real-world bike sharing data.*



