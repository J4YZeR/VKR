from pathlib import Path

import pandas as pd

from utils.ml_data_provider import SectoralDataProvider


class TickerByIndustryDataCreator:
    TICKERS_BY_INDEX_DIRECTORY = '/Users/j4yzer/PycharmProjects/VKR/data/tickers_by_industrial_index'
    INDEX_INDUSTRY = {
        'DJUSAE': 'Aerospace & Defense',
        'DJUSBK': 'Banks',
        'DJUSNS': 'Internet',
        'DJUSFB': 'Food & Beverage',
        'DJUSBT': 'Biotechnology',
        'DJUSMC': 'Health Care Equipment & Services',
        'DJUSIR': 'Insurance',
        'DJUSIM': 'Industrial Metals & Mining',
        'DJUSEN': 'Oil & Gas',
        'DJUSPR': 'Pharmaceuticals',
        'DJUSRT': 'Retail',
        'DJUSSV': 'Software & Computer Services',
        'DJUSTQ': 'Technology Hardware & Equipment',
        'DJUSTL': 'Telecommunications',
        'DJUSSC': 'Semiconductor',
        'DJT': 'Transportation'
    }
    def __init__(self, data_filepath = ''):
        self.data_filepath = data_filepath
    def create_ticker_by_industry_data(self):
        data = pd.DataFrame(columns=['ticker', 'industry', 'index'])
        dir = Path(TickerByIndustryDataCreator.TICKERS_BY_INDEX_DIRECTORY)
        for f in dir.iterdir():
            tickers = pd.read_csv(f)
            index = f.stem
            tickers.insert(loc=1, column='industry', value=TickerByIndustryDataCreator.INDEX_INDUSTRY[index])
            tickers.insert(loc=2, column='index', value=index)
            data = pd.concat([data, tickers])
        data.to_csv(self.data_filepath, index=False)
class TickerBySectorDataCreator:
    TICKERS_BY_INDEX_DIRECTORY = '/Users/j4yzer/PycharmProjects/VKR/data/tickers_by_sectoral_index'
    INDEX_SECTOR = {
        'GSPE':'Energy',
        'SP500-15': 'Materials',
        'SP500-20': 'Industrials',
        'SP500-25': 'Consumer Discretionary',
        'SP500-30': 'Consumer Staples',
        'SP500-35': 'Health Care',
        'SP500-40': 'Financials',
        'SP500-45': 'Information Technology',
        'SP500-50': 'Telecommunication Service',
        'SP500-55': 'Utilities',
        'SP500-60': 'Real Estate',
    }

    def __init__(self, data_filepath = SectoralDataProvider.TICKERS_BY_SECTOR_FILEPATH):
        self.data_filepath = data_filepath

    def create_ticker_by_industry_data(self):
        data = pd.DataFrame(columns=['ticker', 'sector', 'index'])
        dir = Path(TickerBySectorDataCreator.TICKERS_BY_INDEX_DIRECTORY)
        for f in dir.iterdir():
            tickers = pd.read_csv(f)
            tickers['ticker'] = tickers['ticker'].replace(regex='(?<=.)-.+', value='')
            tickers = tickers.drop_duplicates()
            index = f.stem
            tickers.insert(loc=1, column='sector', value=TickerBySectorDataCreator.INDEX_SECTOR[index])
            tickers.insert(loc=1, column='index', value=index)
            data = pd.concat([data, tickers])
        data.to_csv(self.data_filepath, index=False)
if __name__ == '__main__':
    TickerBySectorDataCreator().create_ticker_by_industry_data()