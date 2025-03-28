import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor


ticker_pickle_file_path = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/ticker_data.pkl'
stock_prices_pickle_file_path = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/stock_prices.pkl'
index_weights_pickle_file_path = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/index_weights.pkl'
ml_pickle_file_path = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/ml_weights.pkl'

def extract_features(mom, vol):
    return [mom.mean(), mom.std(), vol.mean(), vol.std()]

class MLModel:

    def __init__(self, factors, data, params):
        self.factors = factors
        self.returns = data.returns
        self.params = params

    def train_model(self):
        X = []
        y = []
        training_end_period = self.params['training_end_period']
        month_ends = self.factors.mom_ranks.index.to_series().groupby(self.factors.mom_ranks.index.to_period("M")).last()
        month_ends = month_ends[month_ends <= training_end_period]

        weight_grid = np.round(np.linspace(0, 1, 11), 2)

        for date in tqdm(month_ends[:-2]):
            try:
                mom = self.factors.mom_ranks.loc[date]
                vol = self.factors.vol_ranks.loc[date]

                # Features: einfache Statistik über die Faktorwerte
                x_features = extract_features(mom, vol)

                next_date = month_ends[month_ends > date].iloc[0]
                next_mask = (self.returns.index > date) & (self.returns.index <= next_date)
                next_returns = self.returns.loc[next_mask]

                best_weight = 0.5
                best_return = -np.inf

                for w in weight_grid:
                    combined_scores = mom * w + vol * (1 - w)
                    combined_scores = combined_scores / combined_scores.sum()
                    perf = next_returns[combined_scores.index].dot(combined_scores)
                    cum_return = (1 + perf).prod()
                    if cum_return > best_return:
                        best_return = cum_return
                        best_weight = w

                X.append(x_features)
                y.append(best_weight)
            except:
                continue

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        with open(ml_pickle_file_path, "wb") as file:
            pickle.dump(model, file)

        print("ML-Modell gespeichert.")

class Data:
    def __init__(self, params: dict):
        self.price_frequency_str = params['price_frequency_str']
        self.use_pickle_data = params['use_pickle_data']

    def get_data(self):

        if self.use_pickle_data:

            with open(stock_prices_pickle_file_path, 'rb') as file:
                stock_prices = pickle.load(file)

            with open(index_weights_pickle_file_path, 'rb') as file:
                index_weights = pickle.load(file)
        else:
            # 1. Ticker Mapping
            ticker_mapping = pd.read_excel(
                r'G:\TEC101\ALLE\Zink\40_CPF Program\Data\Final Project\isin_msci_ticker_mapping.xlsx')
            ticker_mapping.drop(columns=['Unnamed: 0'], inplace=True)
            ticker_mapping['MSCI_SECURITY_CODE'] = ticker_mapping['MSCI_SECURITY_CODE'].astype(str)
            ticker_mapping['BBG_TICKER'] = ticker_mapping['BBG_TICKER'].astype(str)
            mapping_dict = dict(zip(ticker_mapping['MSCI_SECURITY_CODE'], ticker_mapping['BBG_TICKER']))

            # 2. Stock Prices
            stock_prices = pd.read_excel(r'G:\TEC101\ALLE\Zink\40_CPF Program\Data\Final Project\stock_prices.xlsx')
            stock_prices.index = stock_prices['POS_DATE']
            stock_prices.drop(columns=['POS_DATE'], inplace=True)
            if self.price_frequency_str == 'D':
                pass
            else:
                stock_prices = stock_prices.resample(self.price_frequency_str).last()
            stock_prices = stock_prices.rename(columns=mapping_dict)

            with open(stock_prices_pickle_file_path, 'wb') as file:
                pickle.dump(stock_prices, file)

            # 3. Index Weights
            index_weights = pd.read_excel(r'G:\TEC101\ALLE\Zink\40_CPF Program\Data\Final Project\index_weight.xlsx')
            index_weights.index = index_weights['AS_OF_DATE']
            index_weights.drop(columns=['AS_OF_DATE'], inplace=True)
            # index_weights = index_weights.resample(self.frequency).last()
            index_weights = index_weights.rename(columns=mapping_dict)


            with open(index_weights_pickle_file_path, 'wb') as file:
                pickle.dump(index_weights, file)

        self.stock_prices = stock_prices
        self.returns = stock_prices.pct_change()
        self.index_weights = index_weights


class Factors:

    def __init__(self, data: Data, params: dict):
        self.price_frequency_num = params['price_frequency_num']
        self.equal_weighted_stocks = params['equal_weighted_stocks']
        self.index_weights = data.index_weights
        self.stock_prices = data.stock_prices
        self.returns = data.returns
        self.live_begin_period = params['live_begin_period']

    def get_factor_scores(self):
        # 1. Momentum
        # 1. 1. Calculate 12-Month Returns
        perf_12m = self.stock_prices / self.stock_prices.shift(self.price_frequency_num) - 1

        # 1.2. Calculate Momentum Ranks
        momentum_ranks = pd.DataFrame(index=perf_12m.index, columns=perf_12m.columns)
        for date in perf_12m.index:
            row = perf_12m.loc[date]
            try:
                ranks = row.rank(method="first", ascending=False)
                quintiles = pd.qcut(ranks, q=5, labels=[5, 4, 3, 2, 1])
                momentum_ranks.loc[date] = quintiles
            except ValueError:
                # wenn zu viele NaNs in der Zeile sind
                momentum_ranks.loc[date] = np.nan

        # 2. Low-Volatility
        # 2. 1. Calculate Standard Deviation
        vol_12m = self.returns.rolling(self.price_frequency_num).std() * np.sqrt(self.price_frequency_num)

        # 2. 2. Calculate Low-Volatility Ranks
        vol_ranks = pd.DataFrame(index=vol_12m.index, columns=vol_12m.columns)
        for date in vol_12m.index:
            row = vol_12m.loc[date]
            try:
                ranks = row.rank(method="first", ascending=False)
                quintiles = pd.qcut(ranks, q=5, labels=[1, 2, 3, 4, 5])
                vol_ranks.loc[date] = quintiles
            except ValueError:
                # wenn zu viele NaNs in der Zeile sind
                vol_ranks.loc[date] = np.nan


        self.mom_ranks = momentum_ranks
        self.vol_ranks = vol_ranks

    def get_factor_weight(self):
        self.factor_weight = self.predict_factor_weights()

    def predict_factor_weights(self):

        with open(ml_pickle_file_path, "rb") as f:
            model = pickle.load(f)

        weights = pd.DataFrame(index=self.mom_ranks.index, columns=['WEIGHT_MOM', 'WEIGHT_MIN_VOL'])

        relevant_dates = self.mom_ranks.index[self.mom_ranks.index >= self.live_begin_period]

        for date in relevant_dates:
            try:
                mom = self.mom_ranks.loc[date]
                vol = self.vol_ranks.loc[date]

                x_features = extract_features(mom, vol)
                features = np.array([x_features])
                pred = model.predict(features)[0]
                weights.loc[date] = [pred, 1 - pred]
            except:
                weights.loc[date] = [0.5, 0.5]  # Fallback
        return weights

    def get_stock_weights(self):


        # 1. Multiply Factor Scores with Factor Weights
        factors_weighted = self.mom_ranks.mul(self.factor_weight['WEIGHT_MOM'], axis=0) + self.vol_ranks.mul(self.factor_weight['WEIGHT_MIN_VOL'], axis=0)

        # 2. Deal with nan
        row_sums = factors_weighted.sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)

        # 3. Calculate factored stock weights
        stock_weights = factors_weighted.div(row_sums, axis=0)


        if self.equal_weighted_stocks:
            n_assets = stock_weights.notna().sum(axis=1)
            stock_equal_weight = stock_weights.notna().div(n_assets, axis=0)
            stock_equal_weight = stock_equal_weight.where(stock_weights.notna())
            stock_weights = stock_equal_weight.copy(deep=True)

        stock_weights.index = pd.to_datetime(stock_weights.index)
        self.stock_weights = stock_weights.copy(deep=True)

class Backtest:

    def __init__(self, data: Data, factors: Factors, params: dict):
        self.price_frequency_num = params['price_frequency_num']
        self.returns = data.returns
        self.stock_weights = factors.stock_weights
        self.live_begin_period = params['live_begin_period']

    def run_backtest(self):
        returns = self.returns.copy(deep=True)
        returns.index = pd.to_datetime(returns.index)

        month_ends = self.stock_weights.index.to_series().groupby(self.stock_weights.index.to_period("M")).last()
        month_ends = month_ends[month_ends >= self.live_begin_period]
        portfolio_returns = pd.Series(index=returns.index, dtype=float)

        for date in month_ends[:-1]:
            print(date)
            # Sicherstellen, dass für diesen Rebalancing-Tag Gewichte vorliegen
            if date not in self.stock_weights.index:
                continue

            # Hole die Portfolio-Gewichte am Rebalancing-Stichtag
            weights = self.stock_weights.loc[date].dropna()

            # Entferne Aktien, die nicht in den Renditedaten vorkommen
            weights = weights[weights.index.isin(returns.columns)]

            # Wenn keine sinnvollen Gewichte vorhanden sind, überspringen
            if weights.sum() == 0:
                continue

            # Skaliere die Gewichte so, dass sie sich auf 1 summieren
            weights = weights / weights.sum()

            # Definiere das Live-Fenster: 4 Wochen nach dem Rebalancing-Stichtag
            period_start = date + pd.Timedelta(days=1)
            period_end = month_ends[month_ends > date].iloc[0]

            # Filtere wöchentliche Renditen im Live-Fenster
            mask = (returns.index >= period_start) & (returns.index <= period_end)
            period_returns = returns.loc[mask, weights.index]

            # Berechne wöchentliche Portfolio-Renditen durch gewichtete Summe
            perf = period_returns.dot(weights)

            # Speichere die Ergebnisse im Rückgabe-DataFrame
            portfolio_returns.loc[perf.index] = perf

        self.df_portfolio_returns = pd.DataFrame(portfolio_returns[self.live_begin_period:], columns=['BT_RETURN'])
        self.df_portfolio_returns['BT_CUM_RET'] = (1+self.df_portfolio_returns).cumprod()

        # 5.  Calculate Backtest Performance measures
        avg_return = self.df_portfolio_returns["BT_RETURN"].mean() * self.price_frequency_num
        volatility = self.df_portfolio_returns["BT_RETURN"].std() * np.sqrt(self.price_frequency_num)
        sharpe_ratio = avg_return / volatility
        roll_max = self.df_portfolio_returns["BT_CUM_RET"].cummax()
        drawdown = self.df_portfolio_returns["BT_CUM_RET"] / roll_max - 1
        max_drawdown = drawdown.min()

        self.bt_performance = pd.DataFrame({
            "Average Return": [avg_return],
            "Volatility": [volatility],
            "Sharpe Ratio": [sharpe_ratio],
            "Max Drawdown": [max_drawdown]
        }, index=['BT_PERFORMANCE']).T


if __name__ == '__main__':

    # 0. Define Params
    params_ = {
        'price_frequency_str': 'D',
        'use_pickle_data': True,
        'price_frequency_num': 252,
        'equal_weighted_stocks': True,
        'training_end_period' : pd.Timestamp('2099-12-31'),
        'live_begin_period' : pd.Timestamp('2004-01-01')}

    # 1. Load Data
    data_instance = Data(params = params_)
    data_instance.get_data()

    # 2. Get Factored Weights
    factors = Factors(data = data_instance, params = params_)
    factors.get_factor_scores()
    ml_model = MLModel(factors=factors, data = data_instance, params=params_)
    ml_model.train_model()
    factors.get_factor_weight()
    factors.get_stock_weights()

    # 3. Run Backtest
    backtest = Backtest(data = data_instance, factors = factors, params = params_)
    backtest.run_backtest()
    pass

