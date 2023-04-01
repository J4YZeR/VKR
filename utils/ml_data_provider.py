from pathlib import Path

import pandas as pd

from utils.index_utils import YahooFinanceIndexUtils
from utils.raw_data_provider import QuickFSDataProvider


class DataProvider():
    TICKERS = [  # Dow 30
        # 'AXP', No total current assets
        'AMGN',
        'AAPL',
        'BA',
        'CAT',
        'CSCO',
        'CVX',
        # 'GS',
        'HD',
        'HON',
        'IBM',
        'INTC',
        'JNJ',
        'KO',
        # 'JPM',
        'MCD',
        'MMM',
        'MRK',
        'MSFT',
        'NKE',
        'PG',
        # 'TRV',
        'UNH',
        'CRM',
        'VZ',
        'V',
        'WBA',
        'WMT',
        'DIS'
        # 'DOW'
    ]
    DATA_FIELDS = [
        'ticker',
        'date',
        'absoluteReturn',
        'relativeToSPReturn',
        'nextPeriodRelativeToSPReturn',
        'closePrice',
        'dilutedEarningsPerShare',
        'freeCashFlowPerShare',
        'bookValuePerShare',
        'equityRatio',
        'marketCap',
        'priceToEarningsRatio',
        'priceToSalesRatio',
        'priceToFreeCashFlow',
        'priceToBookRatio',
        'enterpriseValueToSalesRatio',
        'currentRatio',
        'roic',
        'netCurrentAssetValuePerShare',
        'ebitdaMargin',
        'netIncomeMargin',
        'returnOnAssets',
        'returnOnEquity'
    ]
    CACHED_DATA_FILENAME = 'ml_data.csv'
    def __init__(self, cache_path):
        self.cache_path = cache_path
    def load_data(self):
        f = Path(self.cache_path + f'/{DataProvider.CACHED_DATA_FILENAME}')
        if not f.is_file():
            self.__reload_and_cache_data()
        return pd.read_csv(f)
    def __reload_and_cache_data(self):
        qfs_dp = QuickFSDataProvider()
        data = pd.DataFrame(columns=[*DataProvider.DATA_FIELDS])
        for ticker in DataProvider.TICKERS:
            f = Path(self.cache_path + f'/{ticker}.csv')
            if not f.is_file():
                raw_ticker_data = qfs_dp.load_data_for_stock(ticker)
                ticker_data = self.__transform_raw_ticker_data(raw_ticker_data, ticker)
                self.__cache_data(ticker_data, f)
            ticker_data = pd.read_csv(f)
            data = data.append(ticker_data)
        data = data.sort_values(['date']) # TODO correct sorting

        self.__cache_data(data, Path(self.cache_path + f'/{DataProvider.CACHED_DATA_FILENAME}.csv'))
        return data
    def __transform_raw_ticker_data(self, df: pd.DataFrame, ticker) -> pd.DataFrame:
        df.insert(loc=0, column='ticker', value=ticker)
        df['nextPeriodRelativeToSPReturn'] = df['relativeToSPReturn'].shift(periods=-1)
        return df[[*DataProvider.DATA_FIELDS]]
    def __cache_data(self, df: pd.DataFrame, file : Path):
        df.to_csv(file, index=False)
class SectoralDataProvider():
    DATA_FIELDS = [
        'ticker',
        'date',
        'sector',
        'sectoralIndex',
        'absoluteReturn',
        'relativeToSPReturn',
        'relativeToSectoralIndexReturn',
        'nextPeriodRelativeToSectoralIndexReturn',
        'closePrice',
        'dilutedEarningsPerShare',
        'freeCashFlowPerShare',
        'bookValuePerShare',
        'equityRatio',
        'marketCap',
        'priceToEarningsRatio',
        'priceToSalesRatio',
        'priceToFreeCashFlow',
        'priceToBookRatio',
        'enterpriseValueToSalesRatio',
        'currentRatio',
        'roic',
        'netCurrentAssetValuePerShare',
        'ebitdaMargin',
        'netIncomeMargin',
        'returnOnAssets',
        'returnOnEquity'
    ]
    TICKERS_BLACKLIST = [
        'CHRD',
        'MATV',

    ]
    TICKERS_BY_SECTOR_FILEPATH = '/Users/j4yzer/PycharmProjects/VKR/data/tickers-by-sector.csv'
    CACHED_DATA_FILENAME = 'sectoral_ml_data.csv'
    def __init__(self, cache_path='/Users/j4yzer/PycharmProjects/VKR/data/sectoral_ml', max_reload_limit=306):
        self.cache_path = cache_path
        self.max_reload_limit = max_reload_limit
    def load_data(self):
        f = Path(self.cache_path + f'/{SectoralDataProvider.CACHED_DATA_FILENAME}')
        if not f.is_file():
            self.__reload_and_cache_data()
        return pd.read_csv(f)
    def __reload_and_cache_data(self):
        qfs_dp = QuickFSDataProvider()
        data = pd.DataFrame(columns=[*SectoralDataProvider.DATA_FIELDS])
        tickers_by_industry = pd.read_csv(Path(SectoralDataProvider.TICKERS_BY_SECTOR_FILEPATH))
        tickers_by_industry = tickers_by_industry[~tickers_by_industry.ticker.isin(SectoralDataProvider.TICKERS_BLACKLIST)]
        industrial_ticker_rows = list(tickers_by_industry[['ticker', 'sector', 'index']].itertuples(index=False, name=None))
        for i, row in enumerate(industrial_ticker_rows):
            if i > self.max_reload_limit:
                break
            ticker, sector, index = row
            f = Path(self.cache_path + f'/{ticker}.csv')
            if not f.is_file():
                raw_ticker_data = qfs_dp.load_data_for_stock(ticker)
                ticker_data = self.__transform_raw_ticker_data(raw_ticker_data, ticker, sector, index)
                self.__cache_data(ticker_data, f)
            ticker_data = pd.read_csv(f)
            data = pd.concat([data,ticker_data])
        data = data.sort_values(['date']) # TODO correct sorting

        self.__cache_data(data, Path(self.cache_path + f'/{SectoralDataProvider.CACHED_DATA_FILENAME}'))
        return data
    def __transform_raw_ticker_data(self, df: pd.DataFrame, ticker, sector, index) -> pd.DataFrame:
        df.insert(loc=0, column='ticker', value=ticker)
        df.insert(loc=1, column='sector', value=sector)
        df.insert(loc=1, column='sectoralIndex', value=index)
        df['relativeToSectoralIndexReturn'] = self.__get_relative_to_sectoral_index_return(df)
        df['nextPeriodRelativeToSectoralIndexReturn'] = df['relativeToSectoralIndexReturn'].shift(periods=-1)
        return df[[*SectoralDataProvider.DATA_FIELDS]]
    def __get_relative_to_sectoral_index_return(slef, df : pd.DataFrame):
        return df['closePrice'].rolling(window=2).apply(lambda rows : SectoralDataProvider.__get_relative_to_sectoral_index_return_local(rows, df))

    @staticmethod
    def __get_relative_to_sectoral_index_return_local(local_rows: pd.Series, global_rows: pd.DataFrame):
        rows = global_rows.loc[local_rows.index]
        sectoral_index = YahooFinanceIndexUtils().load_index(global_rows['sectoralIndex'].array[0])

        prev_q_index_price = sectoral_index.iloc[sectoral_index.index.get_indexer([rows['date'].iloc[0]], method='nearest')][
            'closePrice'].array[0]
        cur_q_index_price = sectoral_index.iloc[sectoral_index.index.get_indexer([rows['date'].iloc[1]], method='nearest')][
            'closePrice'].array[0]
        prev_q_stock_price = rows['closePrice'].iloc[0]
        cur_q_stock_price = rows['closePrice'].iloc[1]
        return (cur_q_stock_price - prev_q_stock_price) / prev_q_stock_price - (
                    cur_q_index_price - prev_q_index_price) / prev_q_index_price
    def __cache_data(self, df: pd.DataFrame, file : Path):
        df.to_csv(file, index=False)


if __name__ == '__main__':
    data = DataProvider(cache_path='/Users/j4yzer/PycharmProjects/VKR/data/ml').load_data()