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

NOTEBOOK_PATH = 'G:/TEC101/ALLE/Zink/40_CPF Program/CPF Final Project/Final Project Results.ipynb'
HTML_OUTPUT_DIR = 'G:/TEC101/ALLE/Zink/40_CPF Program/Final Project Output/'


def initialize_ml_model(params, update_params):
    # 1. Define Params
    params.update(update_params)

    # 2. Load GetData
    data_ = GetData(params=params)
    data_.get_data()
    data_.get_descriptive_stats()

    # 3. Rank Stocks
    stock_scores_ = GetStockScores(data=data_, params=params)
    stock_scores_.get_stock_scores()

    # 4. Train Model
    trained_ml_model_ = TrainMLModel(stock_scores=stock_scores_, data=data_, params=params)
    trained_ml_model_.train_model()

    # 5. Apply Model
    applied_model_ = ApplyMLModel(data=data_, params=params, stock_scores=stock_scores_,
                                  trained_model=trained_ml_model_)
    applied_model_.get_factor_weight()
    applied_model_.get_stock_weights()

    # 6. Run CalculateBacktest
    backtest_ = CalculateBacktest(data=data_, params=params, applied_model=applied_model_)
    backtest_.run_backtest()

    summary_dict = {'params': params,
                    'data': data_,
                    'stock_scores': stock_scores_,
                    'trained_ml_model': trained_ml_model_,
                    'applied_model': applied_model_,
                    'backtest': backtest_}

    return summary_dict

def factor_return(factor_values, label_order, current_index_weights, past_30d_return):
    weights = pd.qcut(factor_values, q=5, labels=label_order).astype(int) * current_index_weights
    weights = weights / weights.sum()
    return_ts = (weights * past_30d_return).sum(axis=1)
    return (1 + return_ts).cumprod().iloc[-1] - 1

def extract_features(mom, vol, roe, rf, vix, past_30d_return, current_index_weights):
    past_index_return_ts = (past_30d_return * current_index_weights).sum(axis=1)
    past_index_return = (1 + past_index_return_ts).cumprod().iloc[-1] - 1
    past_index_vola = past_index_return_ts.std() * np.sqrt(252)

    mom_return = factor_return(mom, [1, 2, 3, 4, 5], current_index_weights, past_30d_return)
    vol_return = factor_return(vol, [5, 4, 3, 2, 1], current_index_weights, past_30d_return)
    roe_return = factor_return(roe, [1, 2, 3, 4, 5], current_index_weights, past_30d_return)

    features = {
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
        'mom_vol_corr': {
            'value': mom.corr(vol),
            'label': 'corr_mom_vol'
        },
        'mom_roe_corr': {
            'value': mom.corr(roe),
            'label': 'corr_mom_roe'
        },
        'rf': {
            'value': rf.iloc[0]/100,
            'label': 'rf'
        },
        'vix': {
            'value': vix.iloc[0]/100,
            'label': 'vix'
        },
        'past_index_return': {
            'value': past_index_return,
            'label': 'past_index_return'
        },
        'past_index_vola': {
            'value': past_index_vola,
            'label': 'past_index_vola'
        }
    }

    df = pd.DataFrame.from_dict(features, orient='index')
    return df

def run_ml_backtests(model_definitions, params):
    results = []
    for i, model_def in enumerate(model_definitions, start=1):
        model_name = model_def['name']
        updated_params = {k: pd.Timestamp(v) if 'period' in k else v
                          for k, v in model_def.items() if k != 'name'}

        print(f"{'='*30}\n🚀 Start {model_name}\n{'='*30}")
        ml_model = initialize_ml_model(params, updated_params)
        specific_ml_results = {
            'Model': model_name,
            'Params': ml_model['params'],
            'GetData': ml_model['data'],
            'GetStockScores': ml_model['stock_scores'],
            'TrainedMLModel': ml_model['trained_ml_model'],
            'AppliedModel': ml_model['applied_model'],
            'BackTest': ml_model['backtest']}

        results.append(specific_ml_results)
        print(f"✅ {model_name} done\n")

    return results


class GetData:
    def __init__(self, params: dict):
        self.price_frequency_str = params['price_frequency_str']
        self.use_pickle_data = params['use_pickle_data']
        self.price_frequency_num = params['price_frequency_num']

        self.stock_prices = None
        self.index_weights = None
        self.roe = None
        self.rf = None
        self.vix = None
        self.returns = None
        self.descriptive_stats = None

    def get_data(self):
        print("GetData is load...")

        if self.use_pickle_data:

            with open(DATA_PICKLE_FILE_PATH, 'rb') as file:
                data_dictionary = pickle.load(file)

        else:
            # 1. Ticker Mapping
            ticker_mapping = pd.read_excel(r'data.xlsx', sheet_name='isin_msci_ticker_mapping')
            ticker_mapping.drop(columns=['Unnamed: 0'], inplace=True)
            ticker_mapping['MSCI_SECURITY_CODE'] = ticker_mapping['MSCI_SECURITY_CODE'].astype(str)
            ticker_mapping['BBG_TICKER'] = ticker_mapping['BBG_TICKER'].astype(str)
            mapping_dict = dict(zip(ticker_mapping['MSCI_SECURITY_CODE'], ticker_mapping['BBG_TICKER']))

            # 2. Stock Prices
            stock_prices = pd.read_excel(r'data.xlsx',
                                         sheet_name='stock_prices')
            stock_prices.index = stock_prices['POS_DATE']
            stock_prices.drop(columns=['POS_DATE'], inplace=True)
            if self.price_frequency_str == 'D':
                pass
            else:
                stock_prices = stock_prices.resample(self.price_frequency_str).last()
            stock_prices = stock_prices.rename(columns=mapping_dict)
            stock_prices = stock_prices.loc[:, stock_prices.columns.isin(mapping_dict.values())]

            # 3. Index Weights
            index_weights = pd.read_excel(r'data.xlsx',
                                          sheet_name='index_weights')
            index_weights.index = index_weights['AS_OF_DATE']
            index_weights.drop(columns=['AS_OF_DATE'], inplace=True)
            index_weights = index_weights.rename(columns=mapping_dict)
            index_weights = index_weights.loc[:, index_weights.columns.isin(mapping_dict.values())]
            bidx = stock_prices.index.unique()
            index_weights = index_weights.reindex(bidx).ffill()
            index_weights = index_weights.div(index_weights.sum(axis=1), axis=0)

            # Return on Equity
            roe_raw = pd.read_excel(r'data.xlsx',
                                    sheet_name='roe')
            roe = roe_raw.pivot(index='AS_OF_DATE', columns='MSCI_SECURITY_CODE', values='ROE')
            roe.index = pd.to_datetime(roe.index, dayfirst=True)
            roe.sort_index(inplace=True)
            roe.columns = roe.columns.astype(str)
            roe = roe.rename(columns=mapping_dict)
            roe = roe.loc[:, roe.columns.isin(mapping_dict.values())]
            roe = roe.ffill().bfill()
            roe = roe[stock_prices.index.min():stock_prices.index.max()]
            roe.columns.name = None

            # Convert to daily dataframe
            bidx = stock_prices.index.unique()
            roe = roe.reindex(bidx).ffill()


            # 4. RF und VIX
            rf_vix = pd.read_excel(r'data.xlsx',
                                   sheet_name='rf_vix')
            rf_vix.index = rf_vix['Date']
            rf_vix.drop(columns=['Date'], inplace=True)
            rf_vix.index = pd.to_datetime(rf_vix.index)

            # 5. Create GetData Dictionary
            data_dictionary = {'stock_prices' : stock_prices,
                               'index_weights' : index_weights,
                               'roe' : roe,
                               'rf_vix' : rf_vix}

            with open(DATA_PICKLE_FILE_PATH, 'wb') as file:
                pickle.dump(data_dictionary, file) # type: ignore


        # self.stock_prices = stock_prices.loc[stock_prices.index.year >= 2018]
        self.stock_prices = data_dictionary['stock_prices']
        self.index_weights = data_dictionary['index_weights']
        self.roe = data_dictionary['roe']
        self.rf = data_dictionary['rf_vix'][['RF']].copy(deep=True)
        self.vix = data_dictionary['rf_vix'][['VIX']].copy(deep=True)
        self.returns = data_dictionary['stock_prices'].pct_change(fill_method=None)


        print("GetData load.")

    def get_descriptive_stats(self):
        returns = self.returns.dropna(how='all', axis=1)

        basic_info = {
            "start_date": returns.index.min(),
            "end_date": returns.index.max(),
            "num_days": returns.shape[0],
            "num_assets": returns.shape[1],
        }

        missing = returns.isna().sum()
        missing_percent = missing / len(returns) * 100

        availability_stats = {
            "min_missing_pct": missing_percent.min(),
            "mean_missing_pct": missing_percent.mean(),
            "max_missing_pct": missing_percent.max(),
            "num_fully_available_assets": np.sum((missing == 0))
        }

        # Return Stats
        stats = returns.describe().T[["mean", "std", "min", "max"]]
        stats["mean"] *= self.price_frequency_num
        stats["std"] *= np.sqrt(self.price_frequency_num)

        # Save in Dictionary
        self.descriptive_stats = {
            "basic_info": basic_info,
            "availability_stats": availability_stats,
            "return_stats": stats,
            "daily_availability": self.stock_prices.notna().sum(axis=1)
        }


class GetStockScores:
    def __init__(self, data: GetData, params: dict):
        self.price_frequency_num = params['price_frequency_num']
        self.params = params
        self.data = data
        self.roe = data.roe
        self.stock_prices = data.stock_prices
        self.rf = data.rf
        self.vix = data.vix
        self.returns = data.returns
        self.update_factor_scores = params['update_factor_scores']
        self.relevant_factors = params['relevant_factors']

        self.mom_ranks = None
        self.vol_ranks = None
        self.roe_ranks = None
        self.perf_12m = None
        self.vol_12m = None
        self.roe_12m = None

    def get_stock_scores(self):
        print("Stock Scores are calculated")

        stock_scores = {}
        if self.update_factor_scores:
            # 1.1 Raw Stock Scores
            perf_12m = self.stock_prices / self.stock_prices.shift(self.price_frequency_num) - 1
            vol_12m = self.returns.rolling(self.price_frequency_num).std() * np.sqrt(self.price_frequency_num)
            roe_12m = self.roe.rolling(self.price_frequency_num).mean()

            # 1.2 Factor ranks dataframes
            mom_ranks = pd.DataFrame(index=perf_12m.index, columns=perf_12m.columns)
            vol_ranks = pd.DataFrame(index=vol_12m.index, columns=vol_12m.columns)
            roe_ranks = pd.DataFrame(index=vol_ranks.index, columns=vol_ranks.columns)

            # 1.3 fill dataframes
            for date in tqdm(vol_12m.index):
                row_mom = perf_12m.loc[date]
                row_vol = vol_12m.loc[date]
                row_roe = roe_12m.loc[date]

                try:
                    ranks_mom = row_mom.rank(method="first", ascending=False)
                    quintiles_mom = pd.qcut(ranks_mom, q=5, labels=[5, 4, 3, 2, 1])
                    mom_ranks.loc[date] = quintiles_mom

                    ranks_vol = row_vol.rank(method="first", ascending=False)
                    quintiles_vol = pd.qcut(ranks_vol, q=5, labels=[1, 2, 3, 4, 5])
                    vol_ranks.loc[date] = quintiles_vol

                    ranks_roe = row_roe.rank(method='first', ascending=False)
                    quintiles_roe = pd.qcut(ranks_roe, q=5, labels=[5, 4, 3, 2, 1])
                    roe_ranks.loc[date] = quintiles_roe

                except ValueError:
                    mom_ranks.loc[date] = np.nan
                    vol_ranks.loc[date] = np.nan
                    roe_ranks.loc[date] = np.nan


            # Save results in Dictionary
            stock_scores = {
                'mom_ranks': mom_ranks,
                'vol_ranks': vol_ranks,
                'roe_ranks': roe_ranks,
                'perf_12m': perf_12m,
                'vol_12m': vol_12m,
                'roe_12m': roe_12m
            }

            with open(FACTOR_SCORES_PICKLE_FILE_PATH, 'wb') as file:
                pickle.dump(stock_scores, file) # type: ignore

        else:
            with open(FACTOR_SCORES_PICKLE_FILE_PATH, 'rb') as file:
                stock_scores = pickle.load(file)

        self.mom_ranks = stock_scores['mom_ranks'].dropna(how='all')
        self.vol_ranks = stock_scores['vol_ranks'].dropna(how='all')
        self.roe_ranks = stock_scores['roe_ranks'].dropna(how='all')

        self.perf_12m = stock_scores['perf_12m'].dropna(how='all')
        self.vol_12m = stock_scores['vol_12m'].dropna(how='all')
        self.roe_12m = stock_scores['roe_12m'].dropna(how='all')

        print("Stock Scores calculated.")


class TrainMLModel:
    def __init__(self, stock_scores: GetStockScores, data: GetData, params: dict):
        self.stock_scores = stock_scores
        self.data = data
        self.params = params
        self.returns = data.returns
        self.rf = data.rf
        self.vix = data.vix
        self.training_start_period = params['training_start_period']
        self.training_end_period = params['training_end_period']
        self.ml_training_factors = params['ml_training_factors']
        self.relevant_factors = params['relevant_factors']

        self.ml_model = None
        self.goodness_of_model = None
        self.df_importance = None
        self.feature_stats = None


    def train_model(self):
        print("Model is trained...")
        x = []
        y = []
        month_ends = self.stock_scores.mom_ranks.index.to_series().groupby(self.stock_scores.mom_ranks.index.
                                                                           to_period("M")).last()
        month_ends = month_ends[month_ends <= self.training_end_period]
        month_ends = month_ends[month_ends >= self.training_start_period]

        weight_grid = np.round(np.linspace(0, 1, 11), 2)
        x_features = pd.DataFrame()
        for date in tqdm(month_ends[:-2]):
            try:
                # Return GetData
                next_date = month_ends[month_ends > date].iloc[0]
                next_mask = (self.returns.index > date) & (self.returns.index <= next_date)
                next_returns = self.returns.loc[next_mask].dropna(axis=1)

                # Valid Assets
                valid_assets = (self.stock_scores.perf_12m.loc[date].dropna().index.
                                intersection(self.stock_scores.vol_12m.loc[date].dropna().index).
                                intersection(self.stock_scores.roe_12m.loc[date].dropna().index).
                                intersection(next_returns.columns))

                if len(valid_assets) == 0:
                    continue

                # To be standardized
                mom = self.stock_scores.perf_12m.loc[date].loc[valid_assets]
                vol = self.stock_scores.vol_12m.loc[date].loc[valid_assets]
                roe = self.stock_scores.roe_12m.loc[date].loc[valid_assets]

                # To not be standardized
                rf = self.rf.loc[date]
                vix = self.vix.loc[date]
                mom_ranks = self.stock_scores.mom_ranks.loc[date].loc[valid_assets]
                vol_ranks = self.stock_scores.vol_ranks.loc[date].loc[valid_assets]
                roe_ranks = self.stock_scores.roe_ranks.loc[date].loc[valid_assets]

                past_30d_returns = self.returns.loc[self.returns.index < date].tail(30)[valid_assets]
                current_index_weights = self.data.index_weights.loc[date].loc[valid_assets]
                next_returns = next_returns[valid_assets]

                # Features: Simple Statistics
                x_features_full = extract_features(mom, vol, roe, rf, vix, past_30d_returns, current_index_weights)
                x_features = x_features_full[x_features_full['label'].isin(self.ml_training_factors)]
                features = x_features['value'].to_numpy()

                best_weight = 0.5
                best_return = -np.inf

                # Calculate Stock Weights using
                relevant = self.relevant_factors
                factor_ranks = {'mom': mom_ranks, 'roe': roe_ranks, 'vol': vol_ranks}

                for w in weight_grid:
                    score_df = pd.DataFrame({f: factor_ranks[f] for f in relevant})
                    combined_score = score_df.dot([w, 1 - w])
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

        if self.params['ml_model'] == 'RandomForestRegressor':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.params['ml_model'] == 'lightgbm':
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        elif self.params['ml_model'] == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
        elif self.params['ml_model'] == 'catboost':
            model = CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
        else:
            raise ValueError(f"Unknown ML-Model-Name: {self.params['ml_model']}")

        model.fit(x, y)
        y_train_pred = model.predict(x)

        # Analyze goodness of model
        r2 = r2_score(y, y_train_pred)
        mse = mean_squared_error(y, y_train_pred)
        mae = mean_absolute_error(y, y_train_pred)

        goodness_of_model = pd.DataFrame({
                                'Metric': ['R² on training', 'MSE on training', 'MAE on training'],
                                'Value': [r2, mse, mae]
                                })

        # Importance Analysis
        importance = model.feature_importances_
        df_importance = pd.DataFrame(importance, index=x_features['label'].to_list(), columns=["Importance"])

        # Feature Scores
        f_scores, p_values = f_regression(x, np.array(y))
        feature_stats = pd.DataFrame({
            'F-Score': pd.Series(f_scores, index=x_features['label'].to_list()),
            'p-Value': pd.Series(p_values, index=x_features['label'].to_list())
        })

        self.ml_model = model
        self.goodness_of_model = goodness_of_model
        self.df_importance = df_importance
        self.feature_stats = feature_stats


class ApplyMLModel:
    def __init__(self, data: GetData, params: dict, stock_scores: GetStockScores, trained_model: TrainMLModel):
        self.params = params
        self.data = data
        self.mom_ranks = stock_scores.mom_ranks
        self.roe_ranks = stock_scores.roe_ranks
        self.vol_ranks = stock_scores.vol_ranks
        self.returns = stock_scores.returns
        self.perf_12m = stock_scores.perf_12m
        self.vol_12m = stock_scores.vol_12m
        self.roe_12m = stock_scores.roe_12m
        self.rf = stock_scores.rf
        self.vix = stock_scores.vix
        self.ml_model = trained_model.ml_model

        self.factor_weight_predicted = None
        self.factor_weight_5050 = None
        self.factor_weight_av = None
        self.df_weight_summary = None
        self.stock_weights_ml = None
        self.stock_weights_5050 = None
        self.stock_weights_ml_av = None
        self.stock_weights_index = None


    def get_factor_weight(self):
        print("ML factor model is applied to get monthly factor weights...")
        # Calculate Factor Weight from ML Model

        weights = pd.DataFrame(index=self.mom_ranks.index, columns=['WEIGHT_FACTOR_1', 'WEIGHT_FACTOR_2'])

        for date in tqdm(self.mom_ranks.index):
            try:
                mom = self.perf_12m.loc[date].dropna()
                vol = self.vol_12m.loc[date].dropna()
                roe = self.roe_12m.loc[date].dropna()
                rf = self.rf.loc[date]
                vix = self.vix.loc[date]

                past_30d_returns = self.returns.loc[self.returns.index < date].tail(30)
                current_index_weights = self.data.index_weights.loc[date]

                x_features_full = extract_features(mom, vol, roe, rf, vix, past_30d_returns, current_index_weights)
                x_features = x_features_full[x_features_full['label'].isin(self.params['ml_training_factors'])]
                features = x_features['value'].to_numpy()
                features = features.reshape(1, -1)

                pred = self.ml_model.predict(features)[0]
                weights.loc[date] = [pred, 1 - pred]
            except ValueError:
                weights.loc[date] = [0.5, 0.5]

        # Save ML Factor Weight
        self.factor_weight_predicted = weights.copy(deep=True)

        # Calculate 50/50 Factor Weight
        self.factor_weight_5050 = self.factor_weight_predicted.where(self.factor_weight_predicted.isna(), 0.5)

        # Average of ML factor weights
        self.factor_weight_av =  self.factor_weight_predicted.copy(deep=True)
        self.factor_weight_av['WEIGHT_FACTOR_1'] = self.factor_weight_av['WEIGHT_FACTOR_1'].where(
                                                            self.factor_weight_av['WEIGHT_FACTOR_1'].isna(),
                                                            self.factor_weight_av['WEIGHT_FACTOR_1'].mean())
        self.factor_weight_av['WEIGHT_FACTOR_2'] = self.factor_weight_av['WEIGHT_FACTOR_2'].where(
                                                            self.factor_weight_av['WEIGHT_FACTOR_2'].isna(),
                                                            1 - self.factor_weight_av['WEIGHT_FACTOR_1'].mean())

        # Summary
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

        print("ML Factor applied.")

    def get_stock_weights(self):
        print("Stock Weights are calculated using factor weights...")
        factor_ranks = {f: getattr(self, f"{f}_ranks") for f in self.params['relevant_factors']}
        stock_ranks_factor_1 = factor_ranks[self.params['relevant_factors'][0]]
        stock_ranks_factor_2 = factor_ranks[self.params['relevant_factors'][1]]

        # Get Stock weights from stock_scores weights by multiplying ranks with factor weights and rescaling
        stock_weights_ml = self.compute_stock_weights(stock_ranks_factor_1, stock_ranks_factor_2,
                                                      self.factor_weight_predicted, scoring_method='factor_integration')
        stock_weights_5050 = self.compute_stock_weights(stock_ranks_factor_1, stock_ranks_factor_2,
                                                        self.factor_weight_5050, scoring_method='factor_integration')
        stock_weights_ml_av = self.compute_stock_weights(stock_ranks_factor_1, stock_ranks_factor_2,
                                                         self.factor_weight_av, scoring_method='factor_integration')
        stock_weights_index = self.compute_stock_weights(stock_ranks_factor_1, stock_ranks_factor_2,
                                                         self.factor_weight_av, scoring_method='index_weights')

        self.stock_weights_index = stock_weights_index.copy(deep=True)
        self.stock_weights_ml = stock_weights_ml.copy(deep=True)
        self.stock_weights_5050 = stock_weights_5050.copy(deep=True)
        self.stock_weights_ml_av = stock_weights_ml_av.copy(deep=True)
        print("Stock Weights calculated.")

    def compute_stock_weights(self, stock_ranks_factor_1, stock_ranks_factor_2, factor_weights, scoring_method):
        index_weights = self.data.index_weights[stock_ranks_factor_1.index.min():stock_ranks_factor_1.index.max()]

        # Integrate factor weights in stock ranks
        scored_weight = stock_ranks_factor_1.mul(factor_weights['WEIGHT_FACTOR_1'], axis=0) + stock_ranks_factor_2.mul(
                                                    factor_weights['WEIGHT_FACTOR_2'], axis=0)

        # Rescale
        if scoring_method == 'index_weights':
            integrated_weight = index_weights[scored_weight.notna()]
        else:
            integrated_weight = index_weights * scored_weight

        integrated_weight = integrated_weight.div(integrated_weight.sum(axis=1), axis=0)
        integrated_weight.index = pd.to_datetime(integrated_weight.index)
        return integrated_weight.copy(deep=True)


class CalculateBacktest:
    def __init__(self, data: GetData, params: dict, applied_model: ApplyMLModel):
        self.returns = data.returns

        self.price_frequency_num = params['price_frequency_num']
        self.test_start_period = params['test_start_period']
        self.test_end_period = params['test_end_period']

        self.stock_weights_ml = applied_model.stock_weights_ml
        self.stock_weights_5050 = applied_model.stock_weights_5050
        self.stock_weights_ml_av = applied_model.stock_weights_ml_av
        self.stock_weights_index = applied_model.stock_weights_index

        self.df_portfolio_returns = None
        self.bt_performance = None

    def run_backtest(self):
        print("Calculate Backtest is started...")
        returns = self.returns.copy(deep=True)
        returns.index = pd.to_datetime(returns.index)

        # Dictionary of Strategies
        strategies = {
            'ML': self.stock_weights_ml,
            '5050': self.stock_weights_5050,
            'AVG_ML': self.stock_weights_ml_av,
            'INDEX': self.stock_weights_index
        }

        month_ends = strategies['ML'].index.to_series().groupby(strategies['ML'].index.to_period("M")).last()
        month_ends = month_ends[month_ends >= self.test_start_period]
        month_ends = month_ends[month_ends <= self.test_end_period]

        portfolio_returns = {name: pd.Series(index=returns[month_ends.min():].index, dtype=float)
                             for name in strategies}

        for date in tqdm(month_ends[:-1]):
            # Get Rebalancing Window
            period_start = date + pd.Timedelta(days=1)
            period_end = month_ends[month_ends > date].iloc[0]
            mask = (returns.index >= period_start) & (returns.index <= period_end)
            period_returns = returns.loc[mask]

            # For each Strategy
            for name, stock_weights in strategies.items():
                if date not in stock_weights.index:
                    continue

                strategy_weights = stock_weights.loc[date].dropna()
                strategy_weights = strategy_weights[strategy_weights.index.isin(period_returns.columns)]
                strategy_returns = period_returns.loc[:, period_returns.columns.isin(strategy_weights.index)]
                if strategy_weights.sum() == 0:
                    continue
                strategy_weights = strategy_weights / strategy_weights.sum()

                perf = strategy_returns[strategy_weights.index].dot(strategy_weights)
                portfolio_returns[name].update(perf.astype(float))

        # Save and Compare
        df_returns = pd.DataFrame(portfolio_returns)

        df_cum = (1 + df_returns).cumprod()
        df_returns['ML_CUM'] = df_cum['ML']
        df_returns['5050_CUM'] = df_cum['5050']
        df_returns['AVG_ML_CUM'] = df_cum['AVG_ML']
        df_returns['INDEX_CUM'] = df_cum['INDEX']

        self.df_portfolio_returns = df_returns.copy(deep=True)

        # 5.  Calculate CalculateBacktest Performance measures
        performance_dict = {}
        for strat in ['ML', '5050', 'AVG_ML', 'INDEX']:
            strat_returns = self.df_portfolio_returns[strat].dropna()
            strat_cum = self.df_portfolio_returns[f"{strat}_CUM"].dropna()

            avg_return = strat_returns.mean() * self.price_frequency_num
            volatility = strat_returns.std() * np.sqrt(self.price_frequency_num)
            sharpe_ratio = avg_return / volatility if volatility != 0 else np.nan

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
        print("Calculate Backtest done.")


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