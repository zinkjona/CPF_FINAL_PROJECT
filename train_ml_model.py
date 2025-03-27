import pickle
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

def train_model(factors, returns):
    X = []
    y = []

    month_ends = factors.mom_ranks.index.to_series().groupby(factors.mom_ranks.index.to_period("M")).last()
    weight_grid = np.round(np.linspace(0, 1, 11), 2)

    for date in tqdm(month_ends[:-2]):
        try:
            mom = factors.mom_ranks.loc[date]
            vol = factors.vol_ranks.loc[date]

            # Features: einfache Statistik über die Faktorwerte
            x_features = [mom.mean(), mom.std(), vol.mean(), vol.std()]

            next_date = month_ends[month_ends > date].iloc[0]
            next_mask = (returns.index > date) & (returns.index <= next_date)
            next_returns = returns.loc[next_mask]

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

    with open("ml_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("ML-Modell gespeichert.")

# Beispiel-Nutzung (außerhalb dieser Funktion):
# train_model(factors, data_instance.stock_prices.pct_change())
