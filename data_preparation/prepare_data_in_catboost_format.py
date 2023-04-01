import sys

import scipy.stats.distributions
from sklearn.model_selection import KFold

from tuning.random_ranker import ShuffleRanker

sys.path.extend(['/Users/j4yzer/PycharmProjects/VKR'])
from utils.ml_data_provider import SectoralDataProvider

from catboost import CatBoostRanker, Pool, MetricVisualizer, cv
from copy import deepcopy
import numpy as np
import os
import numpy as np
import pandas as pd
import tensorflow as tf
data_provider = SectoralDataProvider(cache_path='/Users/j4yzer/PycharmProjects/VKR/data/sectoral_ml')
data: pd.DataFrame = data_provider.load_data()
def format_data():
    data_provider = SectoralDataProvider(cache_path='/Users/j4yzer/PycharmProjects/VKR/data/sectoral_ml')
    data: pd.DataFrame = data_provider.load_data()

    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', utc=False)

    data = data.replace([-np.Inf, np.Inf], np.nan)
    data = data.dropna()
    data_by_sector = {sector: sector_data for sector, sector_data in data.groupby('sector')}
    for sector, sector_data in data_by_sector.items():
        sector_data = sector_data.groupby("date").filter(lambda x: len(x) > 30)
        sector_data = sector_data[
            sector_data.groupby('ticker')['date'].transform('nunique') == sector_data['date'].nunique()]
        sector_data['nextPeriodRelativeToSectoralIndexReturn'] = sector_data.groupby("date")[
            "nextPeriodRelativeToSectoralIndexReturn"].rank("dense", ascending=True).astype(int)
        sector_data.rename(columns={'nextPeriodRelativeToSectoralIndexReturn': 'nextPeriodRank'}, inplace=True)
        sector_data['relativeToSectoralIndexReturn'] = sector_data.groupby('date')[
            'relativeToSectoralIndexReturn'].rank(
            'dense', ascending=True).astype(int)
        sector_data.rename(columns={'relativeToSectoralIndexReturn': 'rank'}, inplace=True)
        data_by_sector[sector] = sector_data
    energy_data = data_by_sector['Energy']

    energy_data['nextPeriodRank'] = energy_data['nextPeriodRank'] / energy_data['nextPeriodRank'].max()
    energy_data['rank'] = energy_data['rank'] / energy_data['rank'].max()
    energy_data: pd.DataFrame = energy_data
    energy_data['qid'] = energy_data['date'].astype('int64')
    return energy_data

def split_data_into_sets(energy_data):
    # train / valid / test data separation
    time_config = {'train': '2000-01-01', 'valid': '2014-01-01', 'test': '2018-01-01'}
    train_energy_data = energy_data[
        (energy_data['date'] > time_config['train']) & (energy_data['date'] <= time_config['valid'])]
    test_energy_data = energy_data[
        (energy_data['date'] > time_config['valid']) & (energy_data['date'] <= time_config['test'])]
    y_train = train_energy_data[['nextPeriodRank']]

    X_train = train_energy_data[train_energy_data.drop(columns=['ticker', 'sector',
                                                                'closePrice',
                                                                'sectoralIndex', 'nextPeriodRank', 'date',
                                                                'qid']).columns]
    queries_train = train_energy_data[['qid']]

    y_test = test_energy_data[['nextPeriodRank']]
    X_test = test_energy_data[test_energy_data.drop(columns=['ticker', 'sector',
                                                             'closePrice',
                                                             'sectoralIndex', 'nextPeriodRank', 'date', 'qid']).columns]
    queries_test = test_energy_data[['qid']]
    return (X_train, y_train, queries_train, X_test, y_test, queries_test)
def make_pools(X_train, y_train, queries_train, X_test, y_test, queries_test):
    from numpy import int64
    from tuning.random_ranker import ShuffleRanker

    train = Pool(
        data=X_train,
        label=y_train,
        group_id=queries_train
    )

    test = Pool(
        data=X_test,
        label=y_test,
        group_id=queries_test
    )
    return (train, test)
default_parameters = {
    'iterations': 100,
    'custom_metric': ['NDCG:top=10', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0
}

parameters = {}

def fit_model(loss_function, train_pool, test_pool, additional_params=None):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = ShuffleRanker(**parameters) if loss_function == 'ShuffleRanker' else CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model

cv_default_parameters = {
    'custom_metric': ['NDCG:top=10', 'PFound', 'AverageGain:top=10'],
    'iterations': 100,
    'verbose': False,
    'random_seed': 0
}

def cv_model(loss_function: str, pool : Pool, additional_params=None, fold_count=5, train_dir_root='cv_info'):
    os.makedirs(train_dir_root, exist_ok=True)
    parameters = deepcopy(cv_default_parameters)
    if additional_params is not None:
        parameters.update(additional_params)

    kf = KFold(n_splits=fold_count)
    train_df = pd.DataFrame(
        np.concatenate((pool.get_features().T, [pool.get_group_id_hash()], [pool.get_label()]), axis=0).T,
        columns=[*pool.get_feature_names(), 'group_id', 'label'])
    X = train_df.drop(['label', 'group_id'], axis=1)
    y = train_df['label']
    group_ids = train_df['group_id'].astype(np.uint64)
    fold_scores = {}
    for fold_num, indexes in enumerate(kf.split(train_df)):
        train_index, test_index = indexes
        print('Fold number', fold_num)
        cur_train_dir = train_dir_root + '/fold_' + fold_num.__str__()
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        gid_train, gid_val = group_ids.iloc[train_index], group_ids.iloc[test_index]

        cur_train_pool = Pool(data=X_train,
                              label=y_train,
                              group_id=gid_train)
        cur_val_pool = Pool(data=X_val,
                            label=y_val,
                            group_id=gid_val)

        model = fit_model(loss_function, additional_params={**parameters, 'train_dir': cur_train_dir},
                          train_pool=cur_train_pool,
                          test_pool=cur_val_pool)
        fold_scores['fold_' + fold_num.__str__()] = model.get_best_score()
        print(model.get_best_score())
        print()
    return fold_scores
def get_train_test_pools():
    return make_pools(*split_data_into_sets(format_data()))