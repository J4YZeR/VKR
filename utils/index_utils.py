from pathlib import Path

import pandas as pd
import yfinance as yf


class SnP500Utils:
    def __init__(self):
        f = Path('/Users/j4yzer/PycharmProjects/VKR/data/snp500.csv')
        if (not f.is_file()):
            self.__reload_snp500().to_csv(f, index=True) # TODO
        self.snp500df = pd.read_csv(f, parse_dates=['Date'], index_col='Date')
    def __reload_snp500(self):

        snp = yf.Ticker("^GSPC")
        snphistory = snp.history(start='2000-01-01', interval='1d')
        snphistory = snphistory.rename(columns={'Close': 'closePrice'})
        snphistory.index = snphistory.index.map(lambda x: x.replace(tzinfo=None))
        return snphistory.filter(items=['date', 'closePrice'])
    def load_snp500(self) -> pd.DataFrame:
        return self.snp500df
class YahooFinanceIndexUtils:
    def __init__(self, cache_path='/Users/j4yzer/PycharmProjects/VKR/data/invest_cache'):
        self.cache_path = cache_path
    def __reload_index(self, index):
        f = Path(self.cache_path + f'/{index}.csv')
        index = yf.Ticker('^' + index)
        indexhistory = index.history(start='2000-01-01', interval='1d')
        indexhistory = indexhistory.rename(columns={'Close': 'closePrice'})
        indexhistory.index = indexhistory.index.map(lambda x: x.replace(tzinfo=None))
        indexhistory.filter(items=['date', 'closePrice']).to_csv(f, index=True)
    def load_index(self, index) -> pd.DataFrame:
        f = Path(self.cache_path + f'/{index}.csv')
        if not f.is_file():
            self.__reload_index(index)
        return pd.read_csv(f, parse_dates=['Date'], index_col='Date')
if __name__ == '__main__':
    # ind = inv.indices.get_indices('united states')
    index = SnP500Utils()