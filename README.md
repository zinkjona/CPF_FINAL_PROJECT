# 📊 Applying Machine Learning to a Multi-Factor Equity Strategy

This project is to test how ML strategies can enhance equity factor investment strategies.

---

## Dynamic Equity Factor Weighting with Machine Learning

Applying ML models to dynamically weight equity factors (momentum, volatility, profitability).

---

## Overview

This project implements a data-driven pipeline to create dynamic, machine-learned blends of equity factors for portfolio construction and backtesting. It integrates classic cross-sectional signals (momentum, volatility, profitability/ROE) with macro features (VIX, risk-free rate) and leverages ML regressors (Random Forest, LightGBM, XGBoost, CatBoost) to infer optimal factor blends on a monthly basis.

---

## Pipeline Steps

1. **Load & Align Data**
   - Price panel, index weights, ROE metrics, risk-free rate and VIX.
2. **Compute Factor Scores & Ranks**
   - Calculate rolling 12M metrics and assign quintile ranks for each factor.
3. **Train ML Regressor**
   - For each period, compute the ex-ante optimal blend weight between two factors and train a regression model to predict these weights from macro/factor features.
4. **Apply Predicted Weights**
   - Use the trained ML model to infer monthly blend weights, build portfolio weights, and generate alternative benchmarks.
5. **Backtest & Report**
   - Simulate realized strategy performance out-of-sample and summarize annualized return, volatility, Sharpe, drawdown etc., including rich visualizations.

---

## Key Features

- **Flexible Data Handling**: Automatic alignment and cleaning of prices, fundamentals, risk-free and macro series.
- **Plug-and-Play ML Models**: Switch between RandomForest, LightGBM, XGBoost, CatBoost.
- **Configurable Scenarios**: Control rebalancing frequency, training/test periods, blend factors, and feature sets via a simple `params` dictionary.
- **Comprehensive Reporting**: Outputs include cumulative return plots, excess return decompositions, feature importances, and performance summaries.

---

## Installation

**Requirements:**
- Python 3.8+
- `numpy`, `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`
- Excel I/O: `openpyxl` or `xlrd` for pandas

Install dependencies with:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm catboost openpyxl