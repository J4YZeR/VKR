import os
from datetime import date
# import fmpsdk
from pathlib import Path

import pandas as pd
import quickfs
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from settings import STOCK_METRICS_WHITELIST
from utils.quickfs_utils import QFSStockMetricsMapper


class QuickFSDataProvider(object):
    def __init__(self):
        load_dotenv()
        self.apikey = os.environ.get("API_QUICKFS")
        self.qfs_smp = QFSStockMetricsMapper()
    def load_data_for_stock(self, ticker: str): # Should return all key metrics + absolute return delta (cur)
        f = Path(f'/Users/j4yzer/PycharmProjects/VKR/data/{ticker}.csv')
        if (not f.is_file()):
            quarterly_stock_fundamentals = self.__get_qsf_from_qfs_source(ticker)
            self.__write_dataframe_to_csv(quarterly_stock_fundamentals, f)
        return self.__read_dataframe_from_csv(f)

    def __get_qsf_from_qfs_source(self, ticker: str):
        qsf_raw = self.__get_qsf_raw(ticker)
        return self.qfs_smp.raw_to_formatted(qsf_raw)
    def __get_qsf_raw(self, ticker: str):
        f = Path(f'/Users/j4yzer/PycharmProjects/VKR/data/quickfs_cache/{ticker}.csv')
        if (not f.is_file()):
            print(ticker)
            info = quickfs.QuickFS(self.apikey).get_data_full(ticker)
            q_info = info['financials']['quarterly']
            cols = list(q_info.keys())
            df = pd.DataFrame(q_info).explode(cols)
            df['period_end_date'] = pd.to_datetime(df['period_end_date'], ).map(lambda x: x + pd.offsets.BMonthEnd())
            df.to_csv(f, index=False)
        return pd.read_csv(f, parse_dates=['period_end_date'])
    def __write_dataframe_to_csv(self, quarterly_stock_fundamentals : pd.DataFrame, f):
        quarterly_stock_fundamentals.to_csv(path_or_buf=f, index=False, mode='x')

    def __read_dataframe_from_csv(self, f):
        with f.open() as csvfile:
            return pd.read_csv(csvfile)

class FmpDataProvider(object):
    def __init__(self):
        load_dotenv()
        self.apikey = os.environ.get("apikey")


    # TODO Refactor: add caching decorator which returns csv if exists or else writes new csv
    def get_quarterly_fundamentals_for_stock(self, ticker: str): # Should return all key metrics + absolute return delta (cur)
        f = Path(f'data/fmp/{ticker}.csv') # TODO
        if (not f.is_file()):
            quarterly_stock_fundamentals = self.get_qsf_from_fmp_source(ticker)

            self.write_dataframe_to_csv(quarterly_stock_fundamentals, f)
        return self.read_dataframe_from_csv(f)
        # Request all key metrics for ticker for period and make quarter data from annual # TODO add period to function params
        # Filter metrics?
        # Request stock price for prev/cur quarter, for every quarter and add cur return # TODO Calculate relative return
        # Return dict



    def read_dataframe_from_csv(self, f: Path):
        with f.open() as csvfile:
            return pd.read_csv(csvfile)

    def annual_to_quarter_list(self, annual: dict):
        start_date = date.fromisoformat(annual['date']) - relativedelta(years=1)
        return [{ **annual, 'date': (start_date + relativedelta(months=3 * i)).isoformat()} for i in range(1,5)]

    def get_metrics_from_fmp_key_metrics(self, key_metrics: pd.DataFrame):
        key_metrics = self.rename_fmp_key_metrics(key_metrics)
        key_metrics = key_metrics.filter(['date', *STOCK_METRICS_WHITELIST])
        return key_metrics
    def rename_fmp_key_metrics(self, key_metrics: pd.DataFrame):
        return key_metrics
    def rename_fmp_financial_ratios(self, financial_ratios: pd.DataFrame):
        return financial_ratios
    def get_metrics_from_fmp_financial_ratios(self, financial_ratios: pd.DataFrame):
        financial_ratios = self.rename_fmp_key_metrics(financial_ratios)
        financial_ratios = financial_ratios.filter(['date', *STOCK_METRICS_WHITELIST])
        return financial_ratios
    def write_dataframe_to_csv(self, df: pd.DataFrame, f: Path):
        with f.open(mode='w', newline='') as csvfile:
            df.to_csv(f, index=False)

    def get_qsf_from_fmp_source(self, ticker: str):
        # ticker_key_metrics = pd.DataFrame(self.annual_to_quarter(fmpsdk.key_metrics(apikey=self.apikey, limit=40, symbol=ticker)))
        # ticker_financial_ratios = pd.DataFrame(self.annual_to_quarter(fmpsdk.financial_ratios(apikey=self.apikey, limit=40, symbol=ticker)))
        # merged_metrics = pd.merge(self.get_metrics_from_fmp_key_metrics(ticker_key_metrics), self.get_metrics_from_fmp_financial_ratios(ticker_financial_ratios))
        #
        # # TODO pull financial ratios (quarterly and aligned with key metrics by date)
        # # TODO merge fr and km and append to every entry
        # #  relative_return=(cur_quarter_price-last_quarter_price)/prev_quarter_price) - ((cur_snp500_price-prev_snp500_price)/prev_snp500_price)
        # return merged_metrics
        pass

    def annual_to_quarter(self, annual_list: list):
        return [quarterly for annual in annual_list for quarterly in self.annual_to_quarter_list(annual)]


if __name__ == '__main__':
    print(FmpDataProvider().get_quarterly_fundamentals_for_stock(ticker='AAPL'))