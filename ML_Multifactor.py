# TODO: optimize model setting (GridSearchCV or RandomizedSearchCV) in xgboost
# TODO: factor weights over time plotting

import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

DATA_PICKLE_FILE_PATH = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/inputs/data.pkl'
FACTOR_SCORES_PICKLE_FILE_PATH = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/outputs/factor_scores.pkl'

def initialize_ml_model(params, update_params):
    # This function coordinates the entire workflow for a machine learning (ML) based backtest,
    # using the provided parameters and also applying any scenario-specific parameter updates.
    # It returns a dictionary containing all objects/results from each step.

    # 1. Update parameters with scenario/model-specific settings
    params.update(update_params)

    # 2. Load all required data and generate descriptive statistics
    data_ = GetData(params=params)
    data_.get_data()                  # Load and process input data
    data_.get_descriptive_stats()     # Calculate descriptive statistics about the data

    # 3. Calculate factor scores for each stock (e.g. momentum, volatility, ROE)
    stock_scores_ = GetStockScores(data=data_, params=params)
    stock_scores_.get_stock_scores()

    # 4. Initialize and train machine learning model to predict factor integration weights
    trained_ml_model_ = TrainMLModel(stock_scores=stock_scores_, data=data_, params=params)
    trained_ml_model_.train_model()

    # 5. Apply trained ML model: compute factor weights and final stock weights
    applied_ml_model_ = ApplyMLModel(data=data_, params=params, stock_scores=stock_scores_,
                                  trained_model=trained_ml_model_)
    applied_ml_model_.get_factor_weight()   # Generate ML-based factor weights (monthly)
    applied_ml_model_.get_stock_weights()   # Calculate final stock weights

    # 6. Run backtest: simulate portfolio performance using the computed weights
    backtest_ = CalculateBacktest(data=data_, params=params, applied_ml_model=applied_ml_model_)
    backtest_.run_backtest()

    # 7. Gather all results in a summary dictionary and return
    summary_dict = {'params': params,
                    'data': data_,
                    'stock_scores': stock_scores_,
                    'trained_ml_model': trained_ml_model_,
                    'applied_ml_model': applied_ml_model_,
                    'backtest': backtest_}

    return summary_dict

def factor_return(factor_values, label_order, current_index_weights, past_30d_return):
    # This function calculates the total return for a factor-based portfolio.
    # It groups stocks into quintiles according to the factor, assigns weights, and computes the return over a period.

    # 1. Assign each stock to a quintile based on its factor value.
    #    The 'label_order' parameter defines the weight ranking (e.g., [1,2,3,4,5] or reverse).
    weights = pd.qcut(factor_values, q=5, labels=label_order).astype(int) * current_index_weights

    # 2. Normalize the weights so they sum to 1.
    weights = weights / weights.sum()

    # 3. Calculate the weighted return time series:
    #    For each date, multiply the assigned weight by the past 30-days returns for each stock, then sum across stocks.
    return_ts = (weights * past_30d_return).sum(axis=1)

    # 4. Calculate the cumulative return over the time series and return the total percentage return.
    return (1 + return_ts).cumprod().iloc[-1] - 1

def extract_features(mom, vol, roe, rf, vix, past_30d_return, current_index_weights):
    # This function computes a variety of aggregated features/statistics from past data
    # for use as input (predictors) in the ML model.

    # 1. Calculate the past index (benchmark) return time series using current index weights.
    past_index_return_ts = (past_30d_return * current_index_weights).sum(axis=1)
    # 2. Calculate total cumulative return of the index over the period.
    past_index_return = (1 + past_index_return_ts).cumprod().iloc[-1] - 1
    # 3. Calculate annualized volatility of the index during the period.
    past_index_vola = past_index_return_ts.std() * np.sqrt(252)

    # 4. Calculate factor portfolio returns using the factor_return function for each factor:
    #    - Momentum quintile returns (ascending: [1,2,3,4,5])
    mom_return = factor_return(mom, [1, 2, 3, 4, 5], current_index_weights, past_30d_return)
    #    - Volatility quintile returns (descending: [5,4,3,2,1])
    vol_return = factor_return(vol, [5, 4, 3, 2, 1], current_index_weights, past_30d_return)
    #    - ROE quintile returns (ascending)
    roe_return = factor_return(roe, [1, 2, 3, 4, 5], current_index_weights, past_30d_return)

    # 5. Collect additional features:
    features = {
        # Historical factor returns
        'past_mom_return': {
            'value': mom_return,
            'label': 'past_mom_return'
        },
        'past_vol_return': {
            'value': vol_return,
            'label': 'past_vol_return'
        },
        'past_roe_return': {
            'value': roe_return,
            'label': 'past_roe_return'
        },
        # Standard deviation (cross-sectional) of factor values
        'momentum_std': {
            'value': mom.std(),
            'label': 'std_mom'
        },
        'volatility_std': {
            'value': vol.std(),
            'label': 'std_vol'
        },
        'roe_std': {
            'value': roe.std(),
            'label': 'std_roe'
        },
        # Cross-sectional correlations between factor scores
        'mom_vol_corr': {
            'value': mom.corr(vol),
            'label': 'corr_mom_vol'
        },
        'mom_roe_corr': {
            'value': mom.corr(roe),
            'label': 'corr_mom_roe'
        },
        # Macro/market data (scaled appropriately)
        'rf': {
            'value': rf.iloc[0]/100,   # Risk-free rate (as decimal instead of percent)
            'label': 'rf'
        },
        'vix': {
            'value': vix.iloc[0]/100,  # VIX volatility index (as decimal instead of percent)
            'label': 'vix'
        },
        # Index summary statistics
        'past_index_return': {
            'value': past_index_return,
            'label': 'past_index_return'
        },
        'past_index_vola': {
            'value': past_index_vola,
            'label': 'past_index_vola'
        }
    }

    # 6. Convert the features dictionary to a DataFrame (row-per-feature, columns: value/label)
    df = pd.DataFrame.from_dict(features, orient='index')
    return df

def run_ml_backtests(model_definitions, params):
    # This function runs multiple ML-based backtests across different scenario/model setups.
    # It applies each configuration, stores detailed results, and prints progress messages.

    results = []

    # Loop over all model definitions (scenarios) one by one
    for i, model_def in enumerate(model_definitions, start=1):
        model_name = model_def['name']

        # Parse scenario parameters: convert those containing 'period' in their key to pandas Timestamps
        updated_params = {k: pd.Timestamp(v) if 'period' in k else v
                          for k, v in model_def.items() if k != 'name'}

        # Print formatted run information
        print(f"{'='*30}\n🚀 Start {model_name}\n{'='*30}")

        # Execute the full ML workflow for the current scenario
        ml_model = initialize_ml_model(params, updated_params)

        # ---- Deltas berechnen ----
        bt_perf = ml_model['backtest'].bt_performance

        ml_val = bt_perf.loc['ML', 'Average Return']
        avg_ml_val = bt_perf.loc['AVG_ML', 'Average Return']
        fivefifty_val = bt_perf.loc['5050', 'Average Return']
        index_val = bt_perf.loc['INDEX', 'Average Return']

        excess = {
            'ML_minus_AVG_ML': ml_val - avg_ml_val if (ml_val is not None and avg_ml_val is not None) else None,
            'ML_minus_5050'  : ml_val - fivefifty_val if (ml_val is not None and fivefifty_val is not None) else None,
            'ML_minus_INDEX' : ml_val - index_val if (ml_val is not None and index_val is not None) else None
        }
        df_excess_return = pd.DataFrame([excess], index=[model_name])

        # --- Extract Feature Stats p-Values
        feature_stats = ml_model['trained_ml_model'].feature_stats
        pvalue_dict = feature_stats['p-Value'].to_dict()
        df_pvalue_final = pd.DataFrame([pvalue_dict], index=[model_name])

        # --- Extract Feature Importance
        imp_stats = ml_model['trained_ml_model'].df_importance
        importance_dict = imp_stats["Importance"].to_dict()
        df_importance_final = pd.DataFrame([importance_dict], index=[model_name])

        # --- Goodness of model
        good_of_model = ml_model['trained_ml_model'].goodness_of_model
        good_of_model = good_of_model.loc[good_of_model['Metric'] == 'R² on training'].copy()
        good_of_model.index = good_of_model['Metric']
        good_of_model = good_of_model.drop(columns=['Metric'])
        gm_dict = good_of_model["Value"].to_dict()
        df_goodness_of_model_final = pd.DataFrame([gm_dict], index=[model_name])

        # --- Average Factor Weights over Time
        avg_factor_weights = ml_model['applied_ml_model'].factor_weight_predicted
        df_avg_factor_weights = pd.DataFrame(avg_factor_weights.mean(), columns= [model_name]).T
        df_avg_factor_weights.columns = ml_model['params']['relevant_factors']

        # Gather all outputs for this model/scenario in a dedicated results dictionary
        specific_ml_results = {
            'Model': model_name,
            'Params': ml_model['params'],
            'GetData': ml_model['data'],
            'GetStockScores': ml_model['stock_scores'],
            'TrainedMLModel': ml_model['trained_ml_model'],
            'AppliedModel': ml_model['applied_ml_model'],
            'BackTest': ml_model['backtest'],
            'ExcessReturns': df_excess_return,
            'FeatureStats': df_pvalue_final,
            'FeatureImportance': df_importance_final,
            'GoodnessOfModel': df_goodness_of_model_final,
            'AverageFactorWeights': df_avg_factor_weights
        }

        # Save the results for further analysis
        results.append(specific_ml_results)
        print(f"✅ {model_name} done\n")

    # Return the list of result dictionaries for all model definitions
    return results


class GetData:
    def __init__(self, params: dict):
        # Initialize the GetData object with relevant parameters from the configuration dictionary

        # Set the price frequency as a string, e.g., 'Monthly', 'Daily', etc.
        self.price_frequency_str = params['price_frequency_str']

        # Specify whether to use pickled (pre-processed) data files
        self.use_pickle_data = params['use_pickle_data']

        # Set the numeric frequency value (e.g., number of days for return calc)
        self.price_frequency_num = params['price_frequency_num']

        # Initialize placeholders for data attributes to be populated later
        self.stock_prices = None         # DataFrame: Historical stock prices
        self.index_weights = None        # DataFrame: Benchmark/index weights per period
        self.roe = None                  # DataFrame: Return on equity values for each stock
        self.rf = None                   # Series: Risk-free rate (per period)
        self.vix = None                  # Series: VIX volatility index (per period)
        self.returns = None              # DataFrame: Calculated returns for all stocks
        self.descriptive_stats = None    # DataFrame/Dict: Descriptive statistics for the dataset

    def get_data(self):
        """
        Loads and processes all required data for analysis and backtesting.
        Data can be loaded either from a pickled file for efficiency or from raw Excel files for full refresh.
        This method updates the class attributes in-place.
        """

        # Check if pickled data should be used (speeds up repeated runs)
        if self.use_pickle_data:
            # Load pre-processed data from a (binary) pickle file
            with open(DATA_PICKLE_FILE_PATH, 'rb') as file:
                data_dictionary = pickle.load(file)

        else:
            # === START: Load and process data from Excel files ===

            # 1. Load ticker mapping: maps security codes (e.g., MSCI codes) to Bloomberg or other ticker formats
            ticker_mapping = pd.read_excel(r'data.xlsx', sheet_name='isin_msci_ticker_mapping')
            ticker_mapping.drop(columns=['Unnamed: 0'], inplace=True)
            ticker_mapping['MSCI_SECURITY_CODE'] = ticker_mapping['MSCI_SECURITY_CODE'].astype(str)
            ticker_mapping['BBG_TICKER'] = ticker_mapping['BBG_TICKER'].astype(str)
            mapping_dict = dict(zip(
                ticker_mapping['MSCI_SECURITY_CODE'],
                ticker_mapping['BBG_TICKER'])
            )

            # 2. Load and process stock price data
            stock_prices = pd.read_excel(r'data.xlsx', sheet_name='stock_prices')
            stock_prices.index = stock_prices['POS_DATE']
            stock_prices.drop(columns=['POS_DATE'], inplace=True)

            # Resample price frequency if not daily. Takes the last data point per period.
            if self.price_frequency_str == 'D':
                pass  # Keep as is (already daily)
            else:
                stock_prices = stock_prices.resample(self.price_frequency_str).last()

            # Rename columns to mapped Bloomberg tickers and keep only tickers present in mapping
            stock_prices = stock_prices.rename(columns=mapping_dict)
            stock_prices = stock_prices.loc[:, stock_prices.columns.isin(mapping_dict.values())]

            # 3. Load and process index weights (benchmark weights per constituent)
            index_weights = pd.read_excel(r'data.xlsx', sheet_name='index_weights')
            index_weights.index = index_weights['AS_OF_DATE']
            index_weights.drop(columns=['AS_OF_DATE'], inplace=True)
            index_weights = index_weights.rename(columns=mapping_dict)
            index_weights = index_weights.loc[:, index_weights.columns.isin(mapping_dict.values())]
            bidx = stock_prices.index.unique()
            # Forward-fill to ensure each price date has weights
            index_weights = index_weights.reindex(bidx).ffill()
            # Normalize weights to sum to 1 for each date
            index_weights = index_weights.div(index_weights.sum(axis=1), axis=0)

            # 4. Load and process Return on Equity (ROE) values
            roe_raw = pd.read_excel(r'data.xlsx', sheet_name='roe')
            roe = roe_raw.pivot(index='AS_OF_DATE', columns='MSCI_SECURITY_CODE', values='ROE')
            roe.index = pd.to_datetime(roe.index, dayfirst=True)
            roe.sort_index(inplace=True)
            roe.columns = roe.columns.astype(str)
            roe = roe.rename(columns=mapping_dict)
            roe = roe.loc[:, roe.columns.isin(mapping_dict.values())]
            # Fill missing values both forward and backward for stability
            roe = roe.ffill().bfill()
            # Restrict ROE data to the date range of stock prices and rename columns
            roe = roe[stock_prices.index.min():stock_prices.index.max()]
            roe.columns.name = None

            # Ensure the ROE index matches stock price index, with forward-fill
            bidx = stock_prices.index.unique()
            roe = roe.reindex(bidx).ffill()

            # 5. Load risk-free rate and VIX volatility index data
            rf_vix = pd.read_excel(r'data.xlsx', sheet_name='rf_vix')
            rf_vix.index = rf_vix['Date']
            rf_vix.drop(columns=['Date'], inplace=True)
            rf_vix.index = pd.to_datetime(rf_vix.index)

            # Combine everything into a dictionary for pickling or direct use
            data_dictionary = {
                'stock_prices': stock_prices,
                'index_weights': index_weights,
                'roe': roe,
                'rf_vix': rf_vix
            }

            # Save to pickle for faster load in future runs (optional)
            with open(DATA_PICKLE_FILE_PATH, 'wb') as file:
                pickle.dump(data_dictionary, file)  # type: ignore

        # === END data loading logic ===

        # Assign loaded data to class attributes for downstream use
        self.stock_prices = data_dictionary['stock_prices']
        self.index_weights = data_dictionary['index_weights']
        self.roe = data_dictionary['roe']
        self.rf = data_dictionary['rf_vix'][['RF']].copy(deep=True)
        self.vix = data_dictionary['rf_vix'][['VIX']].copy(deep=True)
        # Compute returns from prices, using percent change
        self.returns = data_dictionary['stock_prices'].pct_change(fill_method=None)

    def get_descriptive_stats(self):
        """
        Calculates and stores a set of descriptive statistics for the loaded asset returns:
        - Time range, number of assets, missing data percentages, and summary return statistics (annualized).
        - Stores results in self.descriptive_stats.
        """

        # Drop assets entirely missing
        returns = self.returns.dropna(how='all', axis=1)

        # Basic info: time window, data dimensions
        basic_info = {
            "start_date": returns.index.min(),
            "end_date": returns.index.max(),
            "num_days": returns.shape[0],
            "num_assets": returns.shape[1],
        }

        # Asset-level missing data stats
        missing = returns.isna().sum()
        missing_percent = missing / len(returns) * 100
        availability_stats = {
            "min_missing_pct": missing_percent.min(),
            "mean_missing_pct": missing_percent.mean(),
            "max_missing_pct": missing_percent.max(),
            "num_fully_available_assets": np.sum(missing == 0),
        }

        # Return statistics
        stats = returns.describe().T[["mean", "std", "min", "max"]]
        # Annualize mean and std deviation
        stats["mean"] *= self.price_frequency_num
        stats["std"] *= np.sqrt(self.price_frequency_num)

        # Aggregate and save all statistics
        self.descriptive_stats = {
            "basic_info": basic_info,
            "availability_stats": availability_stats,
            "return_stats": stats,
            "daily_availability": self.stock_prices.notna().sum(axis=1)
        }


class GetStockScores:
    def __init__(self, data: GetData, params: dict):
        """
        Initialize the GetStockScores class.

        Arguments:
        - data: an instance of GetData, containing all loaded market/factor data.
        - params: dictionary of config parameters.
        """

        # Frequency number (periods per year), e.g. 252 for daily, 12 for monthly, etc.
        self.price_frequency_num = params['price_frequency_num']
        self.params = params

        # Save reference to input data object (for access to assets/factors/prices)
        self.data = data

        # Expose main dataframes as attributes
        self.roe = data.roe
        self.stock_prices = data.stock_prices
        self.rf = data.rf
        self.vix = data.vix
        self.returns = data.returns

        # Whether to update factor scores dynamically (True: rolling, False: fixed)
        self.update_factor_scores = params['update_factor_scores']

        # Define which model factors to use (e.g. ['mom', 'vol', 'roe'])
        self.relevant_factors = params['relevant_factors']

        # Attributes to hold computed scores/ranks (will be set by scoring methods)
        self.mom_ranks = None  # Momentum ranks
        self.vol_ranks = None  # Volatility ranks
        self.roe_ranks = None  # ROE ranks

        self.perf_12m = None  # 12-month price performance (raw values)
        self.vol_12m = None  # 12-month price volatility (stddev)
        self.roe_12m = None  # 12-month average ROE (or TTM value)

    def get_stock_scores(self):
        """
        Calculates or loads (if previously cached) factor-based stock scores:
          - 12-month price momentum
          - 12-month volatility
          - 12-month mean ROE
        and assigns stocks to quintile ranks for each factor per date.
        Results are stored as attributes for downstream use.
        """

        stock_scores = {}
        # Re-calculate factor scores if requested, else load from cache
        if self.update_factor_scores:
            # ===== 1. Compute rolling raw factor metrics =====
            # 12-month (or N-period) momentum: (current price / price N periods ago) - 1
            perf_12m = self.stock_prices / self.stock_prices.shift(self.price_frequency_num) - 1
            # 12-month volatility: Rolling std dev of returns, annualized
            vol_12m = self.returns.rolling(self.price_frequency_num).std() * np.sqrt(self.price_frequency_num)
            # 12-month average ROE: Rolling mean
            roe_12m = self.roe.rolling(self.price_frequency_num).mean()

            # ===== 2. Initialize empty rank DataFrames =====
            mom_ranks = pd.DataFrame(index=perf_12m.index, columns=perf_12m.columns)
            vol_ranks = pd.DataFrame(index=vol_12m.index, columns=vol_12m.columns)
            roe_ranks = pd.DataFrame(index=vol_12m.index, columns=vol_12m.columns)

            # ===== 3. For each date, assign quintile ranks for each factor =====
            for date in tqdm(vol_12m.index, desc='Calculate Stock Scores'):
                row_mom = perf_12m.loc[date]
                row_vol = vol_12m.loc[date]
                row_roe = roe_12m.loc[date]

                try:
                    # Higher is better for momentum/ROE: rank descending, vol: rank descending (lower vol = higher quintile)
                    ranks_mom = row_mom.rank(method="first", ascending=False)
                    quintiles_mom = pd.qcut(ranks_mom, q=5, labels=[5, 4, 3, 2, 1])
                    mom_ranks.loc[date] = quintiles_mom

                    ranks_vol = row_vol.rank(method="first", ascending=False)
                    quintiles_vol = pd.qcut(ranks_vol, q=5, labels=[1, 2, 3, 4, 5])
                    vol_ranks.loc[date] = quintiles_vol

                    ranks_roe = row_roe.rank(method="first", ascending=False)
                    quintiles_roe = pd.qcut(ranks_roe, q=5, labels=[5, 4, 3, 2, 1])
                    roe_ranks.loc[date] = quintiles_roe

                except ValueError:
                    # If not enough non-NA values to split into quintiles, assign NAs
                    mom_ranks.loc[date] = np.nan
                    vol_ranks.loc[date] = np.nan
                    roe_ranks.loc[date] = np.nan

            # ===== 4. Save for later speed-up =====
            stock_scores = {
                'mom_ranks': mom_ranks,
                'vol_ranks': vol_ranks,
                'roe_ranks': roe_ranks,
                'perf_12m': perf_12m,
                'vol_12m': vol_12m,
                'roe_12m': roe_12m
            }

            with open(FACTOR_SCORES_PICKLE_FILE_PATH, 'wb') as file:
                pickle.dump(stock_scores, file)   # type: ignore

        else:
            # ===== Load from disk (no recalculation) =====
            with open(FACTOR_SCORES_PICKLE_FILE_PATH, 'rb') as file:
                stock_scores = pickle.load(file)

        # ===== 5. Attach as object attributes for future access =====
        self.mom_ranks = stock_scores['mom_ranks'].dropna(how='all')
        self.vol_ranks = stock_scores['vol_ranks'].dropna(how='all')
        self.roe_ranks = stock_scores['roe_ranks'].dropna(how='all')

        self.perf_12m = stock_scores['perf_12m'].dropna(how='all')
        self.vol_12m = stock_scores['vol_12m'].dropna(how='all')
        self.roe_12m = stock_scores['roe_12m'].dropna(how='all')


class TrainMLModel:
    def __init__(self, stock_scores: GetStockScores, data: GetData, params: dict):
        """
        Class to handle machine learning model training on factor/market data.

        Args:
            stock_scores: GetStockScores instance containing factor scores/ranks
            data: GetData instance containing all raw & processed market/factor data
            params: Dictionary of model configuration parameters, e.g.
                - training_start_period: start date for training
                - training_end_period: end date for training
                - ml_training_factors: list of factor names to use as features
                - relevant_factors: factors to use in output/reporting
        """
        self.stock_scores = stock_scores
        self.data = data
        self.params = params

        # Input data (prices, returns, risk-free, VIX)
        self.returns = data.returns
        self.rf = data.rf
        self.vix = data.vix

        # Training configuration
        self.training_start_period = params['training_start_period']
        self.training_end_period = params['training_end_period']
        self.ml_training_factors = params['ml_training_factors']
        self.relevant_factors = params['relevant_factors']

        # Model output attributes (populated after training)
        self.ml_model = None              # Trained model object (e.g. sklearn estimator)
        self.goodness_of_model = None     # Metric(s) like R^2, RMSE, accuracy, etc.
        self.df_importance = None         # Feature importances dataframe (if applicable)
        self.feature_stats = None         # Additional feature statistics (correlations, etc.)

    def train_model(self):
        """
        Trains an ML regressor (RandomForest, LightGBM, XGBoost, CatBoost) to map selected
        factor features to the optimal combination weight for a two-factor strategy.
        For each period, the optimal weight is determined via exhaustive grid search maximizing
        next-period cumulative return.

        Outputs:
            - Trained model (.ml_model property)
            - Model metrics (R², MSE, MAE)
            - Feature importance DataFrame
            - Univariate feature statistics (F-score, p-value)
        """

        x = []  # Feature vectors (per training sample)
        y = []  # Targets: best weights (per sample)

        # 1. Determine eligible training dates (monthly, between start/end)
        month_ends = self.stock_scores.mom_ranks.index.to_series().groupby(
            self.stock_scores.mom_ranks.index.to_period("M")
        ).last()
        month_ends = month_ends[(month_ends >= self.training_start_period) &
                                (month_ends <= self.training_end_period)]

        weight_grid = np.round(np.linspace(0, 1, 11), 2)  # e.g. [0.0, 0.1, ..., 1.0]

        # Placeholder for last extracted feature set (for naming/features)
        x_features = pd.DataFrame()

        # 2. For each eligible training period, determine optimal blend weight by maximizing forward return
        for date in tqdm(month_ends[:-2], desc='Train Model'):  # Exclude last 2 for stability
            try:
                # Next month's endpoints
                next_date = month_ends[month_ends > date].iloc[0]
                next_mask = (self.returns.index > date) & (self.returns.index <= next_date)
                next_returns = self.returns.loc[next_mask].dropna(axis=1)

                # Valid assets: must have all factor values and returns in period
                valid_assets = (
                    self.stock_scores.perf_12m.loc[date].dropna().index
                    .intersection(self.stock_scores.vol_12m.loc[date].dropna().index)
                    .intersection(self.stock_scores.roe_12m.loc[date].dropna().index)
                    .intersection(next_returns.columns)
                )
                if len(valid_assets) == 0:
                    continue

                # Factor raw values (to be standardized)
                mom = self.stock_scores.perf_12m.loc[date].loc[valid_assets]
                vol = self.stock_scores.vol_12m.loc[date].loc[valid_assets]
                roe = self.stock_scores.roe_12m.loc[date].loc[valid_assets]

                # Non-standardized features
                rf = self.rf.loc[date]
                vix = self.vix.loc[date]
                mom_ranks = self.stock_scores.mom_ranks.loc[date].loc[valid_assets]
                vol_ranks = self.stock_scores.vol_ranks.loc[date].loc[valid_assets]
                roe_ranks = self.stock_scores.roe_ranks.loc[date].loc[valid_assets]

                past_30d_returns = self.returns.loc[self.returns.index < date].tail(30)[valid_assets]
                current_index_weights = self.data.index_weights.loc[date].loc[valid_assets]
                next_returns = next_returns[valid_assets]

                # 2a. Feature vector per sample
                x_features_full = extract_features(
                    mom, vol, roe, rf, vix, past_30d_returns, current_index_weights
                )
                # Mask to retain only selected features
                x_features = x_features_full[x_features_full['label'].isin(self.ml_training_factors)]
                features = x_features['value'].to_numpy()

                # 2b. Grid search for optimal blending weight
                best_weight = 0.5
                best_return = -np.inf
                relevant = self.relevant_factors
                factor_ranks = {'mom': mom_ranks, 'roe': roe_ranks, 'vol': vol_ranks}
                for w in weight_grid:
                    score_df = pd.DataFrame({f: factor_ranks[f] for f in relevant})
                    # For two factors: score = w * first + (1-w) * second
                    combined_score = score_df.dot([w, 1 - w])  # Ranks are cross-sectionally assigned
                    combined_score = combined_score.reindex(current_index_weights.index)
                    integrated_weights = current_index_weights * combined_score
                    integrated_weights = integrated_weights / integrated_weights.sum()
                    perf = next_returns.dot(integrated_weights.infer_objects(copy=False).fillna(0))
                    cum_return = (1 + perf).prod()
                    if cum_return > best_return:
                        best_return = cum_return
                        best_weight = w

                x.append(features)
                y.append(best_weight)
            except ValueError:
                continue

        # 3. Choose & fit ML regressor as specified
        ml_type = self.params['ml_model']
        if ml_type == 'RandomForestRegressor':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif ml_type == 'lightgbm':
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        elif ml_type == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
        elif ml_type == 'catboost':
            model = CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
        else:
            raise ValueError(f"Unknown ML-Model-Name: {ml_type}")

        # 4. Train model
        model.fit(x, y)
        y_train_pred = model.predict(x)

        # 5. Model goodness metrics
        r2 = r2_score(y, y_train_pred)
        mse = mean_squared_error(y, y_train_pred)
        mae = mean_absolute_error(y, y_train_pred)
        goodness_of_model = pd.DataFrame({
            'Metric': ['R² on training', 'MSE on training', 'MAE on training'],
            'Value': [r2, mse, mae]
        })

        # 6. Feature importance
        importance = model.feature_importances_
        df_importance = pd.DataFrame(importance, index=x_features['label'].to_list(), columns=["Importance"])

        # 7. Univariate feature scores (F-score, p-value)
        f_scores, p_values = f_regression(x, np.array(y))
        feature_stats = pd.DataFrame({
            'F-Score': pd.Series(f_scores, index=x_features['label'].to_list()),
            'p-Value': pd.Series(p_values, index=x_features['label'].to_list())
        })

        # 8. Save results to instance
        self.ml_model = model
        self.goodness_of_model = goodness_of_model
        self.df_importance = df_importance
        self.feature_stats = feature_stats


class ApplyMLModel:
    def __init__(self, data: GetData, params: dict, stock_scores: GetStockScores, trained_model: TrainMLModel):
        """
        Class to apply a trained ML blending model to current/recent factor and market data.

        Args:
            data: GetData instance (raw data and index weights)
            params: Config dictionary for weighting and strategies
            stock_scores: GetStockScores instance (with factor scores/ranks)
            trained_model: TrainMLModel instance (with .ml_model attribute)
        """

        self.params = params
        self.data = data

        # Store usable factor ranks and values for later portfolio construction
        self.mom_ranks = stock_scores.mom_ranks
        self.roe_ranks = stock_scores.roe_ranks
        self.vol_ranks = stock_scores.vol_ranks
        self.returns = stock_scores.returns
        self.perf_12m = stock_scores.perf_12m
        self.vol_12m = stock_scores.vol_12m
        self.roe_12m = stock_scores.roe_12m

        # Market-wide features
        self.rf = stock_scores.rf      # risk-free rate
        self.vix = stock_scores.vix    # VIX/vol indicator

        # The trained ML model to use for inference
        self.ml_model = trained_model.ml_model

        # Output and summary DataFrames for different strategies
        self.factor_weight_predicted = None   # ML-predicted optimal blend weight (per date)
        self.factor_weight_5050 = None        # Equal weight blend (0.5/0.5)
        self.factor_weight_av = None          # Average learned weight (for benchmarking)

        self.df_weight_summary = None         # All weights in tidy format per date
        self.stock_weights_ml = None          # ML-based portfolio weights (per date/stock)
        self.stock_weights_5050 = None        # 50/50 portfolio weights
        self.stock_weights_ml_av = None       # Portfolio using average weight
        self.stock_weights_index = None       # Benchmark index weights (if needed)

    def get_factor_weight(self):
        """
        Predicts optimal factor blend weights for each period using the trained ML model.
        Stores ML, 50/50, and average weighting schemes and their summary statistics.
        """

        # Prepare DataFrame for factor weights (per date)
        weights = pd.DataFrame(index=self.mom_ranks.index, columns=['WEIGHT_FACTOR_1', 'WEIGHT_FACTOR_2'])

        # Loop over all dates to compute ML-predicted weights
        for date in tqdm(self.mom_ranks.index, desc="ML factor model is applied to get monthly factor weights"):
            try:
                # Each factor: cross-section for this date
                mom = self.perf_12m.loc[date].dropna()
                vol = self.vol_12m.loc[date].dropna()
                roe = self.roe_12m.loc[date].dropna()
                rf = self.rf.loc[date]
                vix = self.vix.loc[date]
                past_30d_returns = self.returns.loc[self.returns.index < date].tail(30)
                current_index_weights = self.data.index_weights.loc[date]

                # Compose feature vector for the current date
                x_features_full = extract_features(mom, vol, roe, rf, vix, past_30d_returns, current_index_weights)
                x_features = x_features_full[x_features_full['label'].isin(self.params['ml_training_factors'])]
                features = x_features['value'].to_numpy().reshape(1, -1)

                # Predict blend weight
                pred = self.ml_model.predict(features)[0]
                # Clamp prediction between 0 and 1 for safety (rare: most regressors remain in bounds)
                pred = max(0, min(1, pred))
                weights.loc[date] = [pred, 1 - pred]

            except ValueError:  # Catch any error (e.g. ValueError, KeyError, etc.) and assign 50/50
                weights.loc[date] = [0.5, 0.5]

        # Store predictions
        self.factor_weight_predicted = weights.copy(deep=True)

        # 50/50 benchmark weights
        self.factor_weight_5050 = self.factor_weight_predicted.where(self.factor_weight_predicted.isna(), 0.5)

        # Average ML factor weight
        factor1_mean = self.factor_weight_predicted['WEIGHT_FACTOR_1'].mean()
        self.factor_weight_av = self.factor_weight_predicted.copy(deep=True)
        self.factor_weight_av['WEIGHT_FACTOR_1'] = self.factor_weight_av['WEIGHT_FACTOR_1'].where(
            self.factor_weight_av['WEIGHT_FACTOR_1'].isna(), factor1_mean)
        self.factor_weight_av['WEIGHT_FACTOR_2'] = self.factor_weight_av['WEIGHT_FACTOR_2'].where(
            self.factor_weight_av['WEIGHT_FACTOR_2'].isna(), 1 - factor1_mean)

        # Create weight summaries for reporting
        summary = {}
        for name, df in {
            'ML': self.factor_weight_predicted,
            '5050': self.factor_weight_5050,
            'AVG_ML': self.factor_weight_av
        }.items():
            summary[name] = {
                'Avg_WEIGHT_FACTOR_1': df['WEIGHT_FACTOR_1'].mean(),
                'Avg_WEIGHT_FACTOR_2': df['WEIGHT_FACTOR_2'].mean(),
                'Count_Observations': df[['WEIGHT_FACTOR_1', 'WEIGHT_FACTOR_2']].notna().all(axis=1).sum()
            }
        self.df_weight_summary = pd.DataFrame(summary).T

    def get_stock_weights(self):
        """
        Calculates stock portfolio weights using various factor blending strategies (ML-predicted blend,
        50/50 blend, average ML blend, and index weights).

        Results are stored as DataFrame attributes for each strategy:
            - self.stock_weights_ml
            - self.stock_weights_5050
            - self.stock_weights_ml_av
            - self.stock_weights_index

        Relies on:
            - self.compute_stock_weights() (should handle weighting/logics as per blend method)
            - factor ranks, factor blend weights present as DataFrames with matching indices
        """

        # Build dictionaries of ranks for each factor of interest
        factor_ranks = {f: getattr(self, f"{f}_ranks") for f in self.params['relevant_factors']}
        factor_1 = self.params['relevant_factors'][0]
        factor_2 = self.params['relevant_factors'][1]
        stock_ranks_factor_1 = factor_ranks[factor_1]
        stock_ranks_factor_2 = factor_ranks[factor_2]

        # Main stock weight constructions
        self.stock_weights_ml = self.compute_stock_weights(
            stock_ranks_factor_1, stock_ranks_factor_2, self.factor_weight_predicted,
            scoring_method='factor_integration')
        self.stock_weights_5050 = self.compute_stock_weights(
            stock_ranks_factor_1, stock_ranks_factor_2, self.factor_weight_5050, scoring_method='factor_integration')
        self.stock_weights_ml_av = self.compute_stock_weights(
            stock_ranks_factor_1, stock_ranks_factor_2, self.factor_weight_av, scoring_method='factor_integration')
        self.stock_weights_index = self.compute_stock_weights(
            stock_ranks_factor_1, stock_ranks_factor_2, self.factor_weight_av, scoring_method='index_weights')

    def compute_stock_weights(self, stock_ranks_factor_1, stock_ranks_factor_2, factor_weights, scoring_method):
        """
        Combines stock factor ranks and factor blending weights to produce a (dates, stocks) DataFrame
        of portfolio weights.

        Args:
            stock_ranks_factor_1: DataFrame of first factor's stock ranks (index=date, columns=asset)
            stock_ranks_factor_2: DataFrame of second factor's stock ranks
            factor_weights: DataFrame with columns ['WEIGHT_FACTOR_1','WEIGHT_FACTOR_2'] (index=date)
            scoring_method:
                - 'factor_integration' : cross-sectionally blend ranks using factor_weights each date, rescale
                - 'index_weights'      : just use index weights on eligible stocks (ignore factor blend)
        Returns:
            integrated_weight: DataFrame (date x stock), normalized rowsum=1 per date
        """

        # Restrict index_weights to the dates in the factor ranks
        date_range = stock_ranks_factor_1.index
        index_weights = self.data.index_weights.reindex(date_range)

        # Score each stock/date by blending the two factor ranks, per supplied factor_weights
        scored_weight = (
                stock_ranks_factor_1.mul(factor_weights['WEIGHT_FACTOR_1'], axis=0) +
                stock_ranks_factor_2.mul(factor_weights['WEIGHT_FACTOR_2'], axis=0)
        )

        # If using 'index_weights', override with raw index weights where available (typically for benchmark port)
        if scoring_method == 'index_weights':
            integrated_weight = index_weights.where(scored_weight.notna())
        else:
            # Otherwise: multiply factor-integration "score" by index wgt
            integrated_weight = index_weights * scored_weight

        # Normalize so each row sums to 1 (portfolio full allocation)
        integrated_weight = integrated_weight.div(integrated_weight.sum(axis=1), axis=0)

        # Final: Ensure index is always datetime, for consistency elsewhere
        integrated_weight.index = pd.to_datetime(integrated_weight.index)

        return integrated_weight.copy(deep=True)  # Defensive copy for downstream safety


class CalculateBacktest:
    def __init__(self, data: GetData, params: dict, applied_ml_model: ApplyMLModel):
        """
        Initialize backtest calculation class.

        Args:
            data (GetData): Contains price/returns/time-series data.
            params (dict): Configuration dictionary with backtest period and settings.
            applied_ml_model (ApplyMLModel): Model instance with precomputed portfolio weights.

        Sets attributes:
            - returns: Returns DataFrame (date x stock)
            - price_frequency_num: Rebalancing frequency (e.g. 21: monthly)
            - test_start_period, test_end_period: Backtest start/end dates
            - stock_weights_ml, stock_weights_5050, stock_weights_ml_av, stock_weights_index:
                  DataFrames of (date x stock) weights for each strategy
            - df_portfolio_returns: To be filled by portfolio return computation
            - bt_performance: To be filled by performance metrics calculation
        """

        # Store asset returns for portfolio calculation
        self.returns = data.returns

        # Backtest timing parameters
        self.price_frequency_num = params['price_frequency_num']
        self.test_start_period = params['test_start_period']
        self.test_end_period = params['test_end_period']

        # Store strategy weights for ML, 50/50, average, and index
        self.stock_weights_ml = applied_ml_model.stock_weights_ml
        self.stock_weights_5050 = applied_ml_model.stock_weights_5050
        self.stock_weights_ml_av = applied_ml_model.stock_weights_ml_av
        self.stock_weights_index = applied_ml_model.stock_weights_index

        # Placeholders for output results (fill with later methods)
        self.df_portfolio_returns = None
        self.bt_performance = None

    def run_backtest(self):
        """
        Run backtest: calculates portfolio returns for each strategy with monthly (or custom) rebalancing,
        then computes performance metrics (average annual return, volatility, Sharpe, max drawdown).

        Stores:
          self.df_portfolio_returns: DataFrame (date x strategies) of realized and cumulative returns.
          self.bt_performance: DataFrame (strategy x metrics) with perf stats.
        """

        # Ensure returns date index is datetime
        returns = self.returns.copy(deep=True)
        returns.index = pd.to_datetime(returns.index)

        # Define all strategies to compare
        strategies = {
            'ML': self.stock_weights_ml,
            '5050': self.stock_weights_5050,
            'AVG_ML': self.stock_weights_ml_av,
            'INDEX': self.stock_weights_index
        }

        # Get month-end (or rebalance-period-end) dates in backtest range
        month_ends = strategies['ML'].index.to_series().groupby(strategies['ML'].index.to_period("M")).last()
        month_ends = month_ends[(month_ends >= self.test_start_period) & (month_ends <= self.test_end_period)]

        # Preallocate portfolio return series for each strategy
        portfolio_returns = {
            name: pd.Series(index=returns.loc[month_ends.min():].index, dtype=float)
            for name in strategies
        }

        # Iterate rebalancing periods
        for date in tqdm(month_ends[:-1], desc='Calculate Backtest'):
            # Define out-of-sample window: after 'date' up to next month end
            period_start = date + pd.Timedelta(days=1)
            period_end = month_ends[month_ends > date].iloc[0]
            mask = (returns.index >= period_start) & (returns.index <= period_end)
            period_returns = returns.loc[mask]

            for name, stock_weights in strategies.items():
                if date not in stock_weights.index:
                    continue

                # Get non-null, tradable stocks at 'date'
                strategy_weights = stock_weights.loc[date].dropna()
                strategy_weights = strategy_weights[strategy_weights.index.isin(period_returns.columns)]
                strategy_returns = period_returns.loc[:, period_returns.columns.isin(strategy_weights.index)]

                # Defensive: skip if all weights are zero (possible for rare corner cases)
                if strategy_weights.sum() == 0:
                    continue

                # Normalize weights (should already sum to 1, but ensures robustness)
                strategy_weights = strategy_weights / strategy_weights.sum()

                # Realized daily portfolio returns
                perf = strategy_returns[strategy_weights.index].dot(strategy_weights)
                portfolio_returns[name].update(perf.astype(float))

        # Build output DataFrame of realized and cumulative returns
        df_returns = pd.DataFrame(portfolio_returns)
        df_cum = (1 + df_returns).cumprod()
        for strat in strategies:
            df_returns[f"{strat}_CUM"] = df_cum[strat]

        self.df_portfolio_returns = df_returns.copy(deep=True)

        # Calculate annualized performance metrics for each strategy
        performance_dict = {}
        for strat in strategies:
            strat_returns = self.df_portfolio_returns[strat].dropna()
            strat_cum = self.df_portfolio_returns[f"{strat}_CUM"].dropna()

            avg_return = strat_returns.mean() * self.price_frequency_num  # Annualized
            volatility = strat_returns.std() * np.sqrt(self.price_frequency_num)
            sharpe_ratio = avg_return / volatility if volatility != 0 else np.nan

            # Max drawdown using cumulative returns
            roll_max = strat_cum.cummax()
            drawdown = strat_cum / roll_max - 1
            max_drawdown = drawdown.min()

            performance_dict[strat] = {
                "Average Return": avg_return,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown": max_drawdown
            }

        self.bt_performance = pd.DataFrame(performance_dict).T
        # print("Calculate Backtest done.")


if __name__ == '__main__':
    params_ = {'use_pickle_data': True,
               'update_factor_scores': False,
               'price_frequency_str': 'D',
               'price_frequency_num': 252,
               'training_start_period': pd.Timestamp('2007-01-31'),
               'training_end_period': pd.Timestamp('2012-12-31'),
               'test_start_period': pd.Timestamp('2013-01-01'),
               'test_end_period': pd.Timestamp('2025-03-20'),
               'ml_training_factors': ['mom_roe_corr', 'past_mom_return', 'past_roe_return',
                                       'past_index_return', 'past_index_vola', 'rf', 'vix'],
               'relevant_factors': ['mom', 'roe'],
               'ml_model': 'RandomForestRegressor'}

    # 2. Set model specifications
    model_definitions_ = [
        dict(name="Model 1",
             training_end_period='2012-12-31',
             test_start_period='2013-01-01',
             ml_training_factors=['mom_roe_corr', 'past_mom_return', 'past_roe_return', 'past_index_return',
                                  'past_index_vola', 'rf', 'vix']),
        dict(name="Model 2",
             training_end_period='2014-12-31',
             test_start_period='2015-01-01',
             ml_training_factors=['mom_roe_corr', 'past_mom_return', 'past_roe_return', 'past_index_return',
                                  'past_index_vola', 'rf', 'vix'])]

    # 3. Run models
    results_ = run_ml_backtests(model_definitions = model_definitions_, params = params_)
    pass