import os

from catboost import CatBoostRanker, Pool
from scipy.stats import randint, loguniform
from sklearn.model_selection import RandomizedSearchCV

import data_preparation.prepare_data_in_catboost_format as data_prep
from tuning.hyperparam_tuning_search import random_search

train, test = data_prep.get_train_test_pools()

loss = 'YetiRankPairwise'

loss_params = {'custom_metric': ['ERR:top=10;hints=skip_train~false','NDCG:top=5;type=Exp;hints=skip_train~false;denominator=LogPosition', 'PrecisionAt:top=10;hints=skip_train~false', 'AverageGain:top=5;hints=skip_train~false', 'MAP:top=10;hints=skip_train~false']}
losses_logdir = {'LambdaMart':'LambdaMart', 'YetiRank':'YetiRank', 'YetiRankPairwise': 'YetiRankPairwise',
                 'ShuffleRanker': 'ShuffleRanker'}
print(loss)
cur_cv_train_dir = 'cv_info/' + losses_logdir[loss]
os.makedirs(cur_cv_train_dir, exist_ok=True)
def get_metrics_without_tuning(pool: Pool, train_dir):
    data_prep.cv_model(loss_function=loss, pool=pool, additional_params=loss_params, fold_count=5, train_dir_root=train_dir)
if __name__ == '__main__':
    grid_distrs = {
        'iterations': randint(50, 1000),
        'learning_rate': loguniform(1e-4, 1e-1),
        'l2_leaf_reg': loguniform(1e-3, 100),
        'depth': randint(1, 7),
        'min_data_in_leaf': randint(4, 14),
    }
    avgs = random_search(loss, cv_pool=train, param_distributions=grid_distrs,
                       iter_num=20, score_metrics=loss_params['custom_metric'], random_seed=42,
                        cv_count=5)
    print('s')