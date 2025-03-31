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
    return [mom.std(), vol.std(), mom.corr(vol)]


def compute_stock_weights(mom_ranks, vol_ranks, factor_weights):
    # Get Stock weights from factors weights by multiplying ranks with factor weights and rescaling

    """
    Berechnet die Aktiengewichte, indem die Faktor-Ranks mit den gegebenen Faktor-Gewichten
    kombiniert und normalisiert werden.
    """
    # Kombiniere die Faktor-Ranks mit den entsprechenden Gewichtungen
    factors_weighted = mom_ranks.mul(factor_weights['WEIGHT_MOM'], axis=0) + vol_ranks.mul(factor_weights['WEIGHT_MIN_VOL'], axis=0)
    row_sums = factors_weighted.sum(axis=1).replace(0, np.nan)
    stock_weights = factors_weighted.div(row_sums, axis=0)
    stock_weights.index = pd.to_datetime(stock_weights.index)
    return stock_weights.copy(deep=True)

class MLModel:

    def __init__(self, factors, data, params):
        self.factors = factors
        self.returns = data.returns
        self.params = params
        self.recalibrate_ml_model = params['recalibrate_ML_model']

    def train_model(self):
        print("Model wird trainiert...")
        if self.recalibrate_ml_model:
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
                    valid_assets = mom.dropna().index.intersection(vol.dropna().index)

                    next_date = month_ends[month_ends > date].iloc[0]
                    next_mask = (self.returns.index > date) & (self.returns.index <= next_date)
                    next_returns = self.returns.loc[next_mask].dropna(axis=1)

                    valid_assets = valid_assets.intersection(next_returns.columns)

                    if len(valid_assets) == 0:
                        continue

                    # Anpassung auf nur valid assets
                    mom = mom.loc[valid_assets]
                    vol = vol.loc[valid_assets]
                    next_returns = next_returns[valid_assets]

                    # Features: einfache Statistik über die Faktorwerte
                    x_features = extract_features(mom, vol)

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
        else:
            print("ML-Modell wird nicht gespeichert.")
        print("Model trainiert.")

class Data:
    def __init__(self, params: dict):
        self.price_frequency_str = params['price_frequency_str']
        self.use_pickle_data = params['use_pickle_data']

    def get_data(self):
        print("Data wird geladen...")

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

        # self.stock_prices = stock_prices.loc[stock_prices.index.year >= 2018]
        self.stock_prices = stock_prices.ffill()
        self.returns = self.stock_prices.pct_change(fill_method=None)
        self.index_weights = index_weights

        print("Data geladen.")


class Factors:

    def __init__(self, data: Data, params: dict):
        self.price_frequency_num = params['price_frequency_num']
        self.index_weights = data.index_weights
        self.stock_prices = data.stock_prices
        self.returns = data.returns
        self.live_begin_period = params['live_begin_period']

    def get_factor_scores(self):
        print("Factor Ränge werden berechnet...")
        # 1.1 Raw Factor Scores
        perf_12m = self.stock_prices / self.stock_prices.shift(self.price_frequency_num) - 1
        vol_12m = self.returns.rolling(self.price_frequency_num).std() * np.sqrt(self.price_frequency_num)

        # 1.2 Factor ranks dataframes
        momentum_ranks = pd.DataFrame(index=perf_12m.index, columns=perf_12m.columns)
        vol_ranks = pd.DataFrame(index=vol_12m.index, columns=vol_12m.columns)

        # 1.3 fill dataframes
        for date in vol_12m.index:
            row_mom = perf_12m.loc[date]
            row_vol = vol_12m.loc[date]
            try:
                ranks_mom = row_mom.rank(method="first", ascending=False)
                quintiles_mom = pd.qcut(ranks_mom, q=5, labels=[5, 4, 3, 2, 1])
                momentum_ranks.loc[date] = quintiles_mom

                ranks_vol = row_vol.rank(method="first", ascending=False)
                quintiles_vol = pd.qcut(ranks_vol, q=5, labels=[1, 2, 3, 4, 5])
                vol_ranks.loc[date] = quintiles_vol
            except ValueError:
                # wenn zu viele NaNs in der Zeile sind
                momentum_ranks.loc[date] = np.nan
                vol_ranks.loc[date] = np.nan

        momentum_ranks.dropna(how='all', inplace=True)
        vol_ranks.dropna(how='all', inplace=True)

        self.mom_ranks = momentum_ranks
        self.vol_ranks = vol_ranks

        print("Factor Ränge berechnet.")


    def get_factor_weight(self):
        print("ML factor model wird angewendet, um Gewichte zu erhalten...")
        # Calculate Factor Weight from ML Model

        with open(ml_pickle_file_path, "rb") as f:
            model = pickle.load(f)

        weights = pd.DataFrame(index=self.mom_ranks.index, columns=['WEIGHT_MOM', 'WEIGHT_MIN_VOL'])
        relevant_dates = self.mom_ranks.index[self.mom_ranks.index >= self.live_begin_period]

        for date in relevant_dates:
            try:
                mom = self.mom_ranks.loc[date].dropna()
                vol = self.vol_ranks.loc[date].dropna()

                x_features = extract_features(mom, vol)
                features = np.array([x_features])
                pred = model.predict(features)[0]
                weights.loc[date] = [pred, 1 - pred]
            except:
                weights.loc[date] = [0.5, 0.5]

        # Save ML Factor Weight
        self.factor_weight = weights.copy(deep=True)

        # Calculate 50/50 Factor Weight
        self.factor_weight_5050 = self.factor_weight.where(self.factor_weight.isna(), 0.5)

        # Average of ML factor weights
        self.factor_weight_av =  self.factor_weight.copy(deep=True)
        self.factor_weight_av['WEIGHT_MOM'] = self.factor_weight_av['WEIGHT_MOM'].where(self.factor_weight_av['WEIGHT_MOM'].isna(), self.factor_weight_av['WEIGHT_MOM'].mean())
        self.factor_weight_av['WEIGHT_MIN_VOL'] = self.factor_weight_av['WEIGHT_MIN_VOL'].where(self.factor_weight_av['WEIGHT_MIN_VOL'].isna(), 1 - self.factor_weight_av['WEIGHT_MOM'].mean())

        # Summary
        summary = {}

        for name, df in {
            'ML': self.factor_weight,
            '5050': self.factor_weight_5050,
            'AVG_ML': self.factor_weight_av
        }.items():
            summary[name] = {
                'Avg_WEIGHT_MOM': df['WEIGHT_MOM'].mean(),
                'Avg_WEIGHT_MIN_VOL': df['WEIGHT_MIN_VOL'].mean(),
                'Count_Observations': df[['WEIGHT_MOM', 'WEIGHT_MIN_VOL']].notna().all(axis=1).sum()
            }

        self.df_weight_summary = pd.DataFrame(summary).T

        print("ML Factor angewendet.")

    def get_stock_weights(self):
        print("Stock Weights werden aus Faktogewichten berechnet...")
        # Get Stock weights from factors weights by multiplying ranks with factor weights and rescaling
        self.stock_weights_ml = compute_stock_weights(self.mom_ranks, self.vol_ranks, self.factor_weight)
        self.stock_weights_5050 = compute_stock_weights(self.mom_ranks, self.vol_ranks, self.factor_weight_5050)
        self.stock_weights_ml_av = compute_stock_weights(self.mom_ranks, self.vol_ranks, self.factor_weight_av)
        print("Stock Weights berechnet.")


class Backtest:

    def __init__(self, data: Data, factors: Factors, params: dict):
        self.price_frequency_num = params['price_frequency_num']
        self.returns = data.returns
        self.live_begin_period = params['live_begin_period']

        self.stock_weights_ml = factors.stock_weights_ml
        self.stock_weights_5050 = factors.stock_weights_5050
        self.stock_weights_ml_av = factors.stock_weights_ml_av


    def run_backtest(self):
        print("Backtest wird gestartet...")
        returns = self.returns.copy(deep=True)
        returns.index = pd.to_datetime(returns.index)

        # Dictionary der Strategien
        strategies = {
            'ML': self.stock_weights_ml,
            '5050': self.stock_weights_5050,
            'AVG_ML': self.stock_weights_ml_av
        }
        portfolio_returns = {name: pd.Series(index=returns.index, dtype=float) for name in strategies}

        month_ends = strategies['ML'].index.to_series().groupby(strategies['ML'].index.to_period("M")).last()
        month_ends = month_ends[month_ends >= self.live_begin_period]

        for date in month_ends[:-1]:
            print(date)
            # Bestimme das Rebalancing-Fenster
            period_start = date + pd.Timedelta(days=1)
            period_end = month_ends[month_ends > date].iloc[0]
            mask = (returns.index >= period_start) & (returns.index <= period_end)
            period_returns = returns.loc[mask]

            # Für jede Strategie:
            for name, stock_weights in strategies.items():
                if date not in stock_weights.index:
                    continue

                weights = stock_weights.loc[date].dropna()
                weights = weights[weights.index.isin(returns.columns)]
                if weights.sum() == 0:
                    continue
                weights = weights / weights.sum()

                perf = period_returns[weights.index].dot(weights)
                portfolio_returns[name].update(perf.astype(float))

        # Abspeichern und Vergleichen
        df_returns = pd.DataFrame(portfolio_returns)
        df_returns = df_returns[self.live_begin_period:]

        df_cum = (1 + df_returns).cumprod()
        df_returns['ML_CUM'] = df_cum['ML']
        df_returns['5050_CUM'] = df_cum['5050']
        df_returns['AVG_ML_CUM'] = df_cum['AVG_ML']

        self.df_portfolio_returns = df_returns.copy(deep=True)

        # 5.  Calculate Backtest Performance measures
        performance_dict = {}
        for strat in ['ML', '5050', 'AVG_ML']:
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
        print("Backtest abgeschlossen.")


if __name__ == '__main__':

    # 0. Define Params
    params_ = {
        'price_frequency_str': 'D',
        'use_pickle_data': True,
        'price_frequency_num': 252,
        'training_end_period' : pd.Timestamp('2099-12-31'),
        'live_begin_period' : pd.Timestamp('2004-01-01'),
        'recalibrate_ML_model' : False}

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

