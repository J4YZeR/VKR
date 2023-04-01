import sys

import scipy.stats.distributions
from sklearn.model_selection import KFold

sys.path.extend(['/Users/j4yzer/PycharmProjects/VKR'])
from utils.ml_data_provider import SectoralDataProvider

from catboost import CatBoostRanker, Pool, MetricVisualizer, cv
from copy import deepcopy
import numpy as np
import os
import numpy as np
import pandas as pd
import tensorflow as tf


#%% md
# Catboost ranking demo
#%%
from catboost.datasets import msrank_10k

train_df, test_df = msrank_10k()

X_train = train_df.drop([0, 1], axis=1).values
y_train = train_df[0].values
queries_train = train_df[1].values
X_test = test_df.drop([0, 1], axis=1).values
y_test = test_df[0].values
queries_test = test_df[1].values
from collections import Counter

Counter(y_train).items()

#%%
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
default_parameters = {
    'iterations': 100,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0
}

parameters = {}


def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model
#%%

model = fit_model('YetiRankPairwise',
                  {'custom_metric': ['NDCG', 'PrecisionAt:top=10', 'RecallAt:top=10', 'MAP:top=10']})
#%%
model
#%% md
# Stock ranking using YetiRankPairwise
#%%
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
    sector_data['relativeToSectoralIndexReturn'] = sector_data.groupby('date')['relativeToSectoralIndexReturn'].rank(
        'dense', ascending=True).astype(int)
    sector_data.rename(columns={'relativeToSectoralIndexReturn': 'rank'}, inplace=True)
    data_by_sector[sector] = sector_data
energy_data = data_by_sector['Energy']

energy_data['nextPeriodRank'] = energy_data['nextPeriodRank'] / energy_data['nextPeriodRank'].max()
energy_data['rank'] = energy_data['rank'] / energy_data['rank'].max()
energy_data: pd.DataFrame = energy_data
energy_data['qid'] = energy_data['date'].astype('int64')

time_config = {'train': '2000-01-01', 'valid': '2014-01-01', 'test': '2018-01-01'}
train_energy_data = energy_data[
    (energy_data['date'] > time_config['train']) & (energy_data['date'] <= time_config['valid'])]
test_energy_data = energy_data[
    (energy_data['date'] > time_config['valid']) & (energy_data['date'] <= time_config['test'])]
#%%
y_train = train_energy_data[['nextPeriodRank']]
X_train = train_energy_data[train_energy_data.drop(columns=['ticker', 'sector',
                                                            'closePrice',
                                                            'sectoralIndex', 'nextPeriodRank', 'date', 'qid']).columns]
queries_train = train_energy_data[['qid']]

y_test = test_energy_data[['nextPeriodRank']]
X_test = test_energy_data[test_energy_data.drop(columns=['ticker', 'sector',
                                                         'closePrice',
                                                         'sectoralIndex', 'nextPeriodRank', 'date', 'qid']).columns]
queries_test = test_energy_data[['qid']]

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
default_parameters = {
    'iterations': 100,
    'custom_metric': ['NDCG:top=10', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0
}

parameters = {}


def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model


cv_default_parameters = {
    'iterations': 100,
    'custom_metric': ['NDCG:top=10', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0
}


def cv_model(loss_function, additional_params=None, fold_count=5, pool=train, train_dir_root='cv_info'):
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

        model = fit_model(loss_function, {**parameters, 'train_dir': cur_train_dir}, train_pool=cur_train_pool, test_pool=cur_val_pool)
        print(model.get_best_score())
        print()
from utils.ranking_eval_utils import get_metrics
from scipy.stats import loguniform, uniform, geom, logser, randint

loss = 'YetiRank'
grid_distrs = {
    'iterations': randint(100, 1000),
    'learning_rate': loguniform(1e-4, 1),
    'l2_leaf_reg': loguniform(1e-3, 100),
    'depth': randint(3, 9),
    'min_data_in_leaf': randint(8, 20),

}
model = CatBoostRanker(loss_function=loss, train_dir=loss, verbose=False, random_seed=2)
model.randomized_search(grid_distrs, X=train, cv=5, search_by_train_test_split=False, plot=False, n_iter=5, calc_cv_statistics=False, refit=False)