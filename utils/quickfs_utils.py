import os
from pathlib import Path

import numpy as np
import pandas as pd
import quickfs
from dotenv import load_dotenv

from settings import STOCK_METRICS_WHITELIST
from utils.index_utils import SnP500Utils


class QFSStockMetricsMapper:
    snp500 = SnP500Utils().load_snp500()
    QFS_TO_STOCK_MARKET_DICT_RENAME = {
        'period_end_date': 'date',
        'period_end_price': 'closePrice',
        'fcf_per_share':'freeCashFlowPerShare',
        'book_value_per_share':'bookValuePerShare',
        'equity_to_assets':'equityRatio',
        'market_cap':'marketCap',
        'price_to_earnings':'priceToEarningsRatio',
        'price_to_sales': 'priceToSalesRatio',
        'price_to_fcf': 'priceToFreeCashFlow',
        'price_to_book': 'priceToBookRatio',
        'enterprise_value_to_sales': 'enterpriseValueToSalesRatio',
        'current_ratio': 'currentRatio',
        'net_income_margin': 'netIncomeMargin',
        'ebitda_margin':'ebitdaMargin',
        'roa': 'returnOnAssets',
        'roe': 'returnOnEquity'
        # 'price_to_earnings_growth': 'priceEarningsToGrowthRatio'

    }
    QFS_TO_STOCK_MARKET_DICT_CREATE = {
        'relativeToSPReturn': lambda df: df['period_end_price'].rolling(window=2).apply(lambda ser: QFSStockMetricsMapper.get_relative_to_snp500_return(ser=ser, df=df)),
        'grahamNumber': lambda x : QFSStockMetricsMapper.get_graham_number(x), # TODO can be NaN because of sqrt(negative_eps * positive_bvps)
        'netCurrentAssetValuePerShare': lambda x: (x['total_current_assets'] - x['total_liabilities']) / x['shares_eop'],
        'absoluteReturn': lambda x: x['period_end_price'].rolling(window=2).apply(lambda x: (x[1] - x[0]) / x[0], raw=True),
        'dilutedEarningsPerShare': lambda x: (x['net_income'] - x['preferred_dividends']) / (x['shares_diluted'])
    }

    @staticmethod
    def get_graham_number(x : pd.DataFrame):
        return np.sqrt(22.5 * ((x['net_income'] - x['preferred_dividends']) / x['shares_basic']) * x['book_value_per_share'])

    @staticmethod
    def get_graham_number_for_row(x: pd.Series):
        res = np.sqrt(22.5 * ((x['net_income'] - x['preferred_dividends']) / x['shares_basic']) * x['book_value_per_share'])
        return res

    @staticmethod
    def get_relative_to_snp500_return(ser: pd.Series, df : pd.DataFrame):
        rows = df.loc[ser.index]
        snp500 = SnP500Utils().load_snp500()

        prev_q_snp500_price = snp500.iloc[snp500.index.get_indexer([rows['period_end_date'].iloc[0]], method='nearest')]['closePrice'].array[0]
        cur_q_snp500_price = snp500.iloc[snp500.index.get_indexer([rows['period_end_date'].iloc[1]], method='nearest')]['closePrice'].array[0]
        prev_q_stock_price = rows['period_end_price'].iloc[0]
        cur_q_stock_price = rows['period_end_price'].iloc[1]
        return (cur_q_stock_price - prev_q_stock_price) / prev_q_stock_price - (cur_q_snp500_price - prev_q_snp500_price) / prev_q_snp500_price

    def raw_to_formatted(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        stock_df = stock_df.sort_values(by='period_end_date')
        for col in (self.QFS_TO_STOCK_MARKET_DICT_CREATE):
            fun = self.QFS_TO_STOCK_MARKET_DICT_CREATE[col]
            stock_df[col] = fun(stock_df)
        stock_df = stock_df.rename(columns=self.QFS_TO_STOCK_MARKET_DICT_RENAME)
        stock_df = stock_df.filter(['date', *STOCK_METRICS_WHITELIST])
        return stock_df

if __name__ == '__main__':
    f = Path('/Users/j4yzer/PycharmProjects/VKR/data/AAPL_qf.csv')
    if (not f.is_file()):
        load_dotenv()
        apikey = os.environ.get('API_QUICKFS')
        info = quickfs.QuickFS(apikey).get_data_full('AAPL')
        q_info = info['financials']['quarterly']
        cols = list(q_info.keys())
        df = pd.DataFrame(q_info).explode(cols)
        df['period_end_date'] = pd.to_datetime(df['period_end_date'], ).map(lambda x: x + pd.offsets.BMonthEnd())
        df.to_csv(f, index=False)
    df = pd.read_csv(f, parse_dates=['period_end_date'])
    f =  Path('/Users/j4yzer/PycharmProjects/VKR/data/quickfs/AAPL.csv')
    QFSStockMetricsMapper().raw_to_formatted(df).to_csv(f, index=False)
