import pandas as pd
import numpy as np
import pickle

ticker_pickle_file_path = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/ticker_data.pkl'
stock_prices_pickle_file_path = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/stock_prices.pkl'
index_weights_pickle_file_path = 'G:/TEC101/ALLE/Zink/40_CPF Program/Data/Final Project/index_weights.pkl'

class Data:
    def __init__(self, frequency: str = 'D', use_pickle_data: bool = True):
        self.frequency = frequency
        self.use_pickle_data = use_pickle_data

    def get_data(self):

        if self.use_pickle_data:
            with open(ticker_pickle_file_path, 'rb') as file:
                ticker_mapping = pickle.load(file)

            with open(stock_prices_pickle_file_path, 'rb') as file:
                stock_prices = pickle.load(file)

            with open(index_weights_pickle_file_path, 'rb') as file:
                index_weights = pickle.load(file)
        else:
            # 1. Ticker Mapping
            ticker_mapping = pd.read_excel(
                r'G:\TEC101\ALLE\Zink\40_CPF Program\Data\Final Project\isin_msci_ticker_mapping.xlsx')
            ticker_mapping.drop(columns=['Unnamed: 0'], inplace=True)

            with open(ticker_pickle_file_path, 'wb') as file:
                pickle.dump(ticker_mapping, file)

            # 2. Stock Prices
            stock_prices = pd.read_excel(r'G:\TEC101\ALLE\Zink\40_CPF Program\Data\Final Project\stock_prices.xlsx')
            stock_prices.index = stock_prices['POS_DATE']
            stock_prices.drop(columns=['POS_DATE'], inplace=True)
            stock_prices = stock_prices.resample(self.frequency).last()

            with open(stock_prices_pickle_file_path, 'wb') as file:
                pickle.dump(stock_prices, file)

            # 3. Index Weights
            index_weights = pd.read_excel(r'G:\TEC101\ALLE\Zink\40_CPF Program\Data\Final Project\index_weight.xlsx')
            index_weights.index = index_weights['AS_OF_DATE']
            index_weights.drop(columns=['AS_OF_DATE'], inplace=True)
            index_weights = index_weights.resample(self.frequency).last()

            with open(index_weights_pickle_file_path, 'wb') as file:
                pickle.dump(index_weights, file)

        return ticker_mapping, stock_prices, index_weights



if __name__ == '__main__':
    frequency_ = 'W-FRI'
    data = Data(frequency=frequency_, use_pickle_data=True)
    ticker_mapping, stock_prices, index_weights = data.get_data()
    pass
    pass
