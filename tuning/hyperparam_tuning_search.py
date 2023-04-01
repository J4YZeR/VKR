import random
from functools import reduce

from scipy.stats import randint, loguniform
from scipy.stats._distn_infrastructure import rv_generic

import data_preparation.prepare_data_in_catboost_format as data_prep_cb


def random_search(loss_function, cv_pool, param_distributions, iter_num=20, random_seed=42, cv_count=5,
                  score_metrics=None):
    random_states = randint(0, 100000).rvs(size=iter_num, random_state=random_seed)
    avg_fold_scores = []
    for num, random_state in enumerate(random_states):
        print('Iteration number', num)
        print(iter_num - num + 1, 'iterations left')
        sampled_params = {key: val.rvs(random_state=random_state) for key, val in param_distributions.items()}
        print(sampled_params)
        fold_scores: dict = data_prep_cb.cv_model(loss_function, cv_pool,
                                                  additional_params={**sampled_params, 'custom_metric': score_metrics},
                                                  fold_count=cv_count,
                                                  train_dir_root='random_search')
        avg_fold_score = reduce(lambda x, y: avg_of_scores(x, y), fold_scores.values())
        print('Average fold score: ', avg_fold_score)
        avg_fold_scores.append((sampled_params, avg_fold_score))
    return avg_fold_scores
def avg_of_scores(x: dict, y: dict):
    avg_score = {}
    for set in x.keys():
        avg_score[set] = {x_key: ((x[set][x_key] + y[set][x_key]) / 2) for x_key
                              in (x[set].keys())}
    return avg_score


if __name__ == '__main__':
    metrics_test_data = {'learn': {'MAP:top=10': 0.6316904761904762, 'ERR:top=10': 0.8807051087753633, 'PrecisionAt:top=10': 0.7485714285714286, 'AverageGain:top=5': 0.7199999998722758, 'NDCG:top=5;type=Exp': 0.714736128592955}, 'validation': {'PFound': 0.889366133028657, 'MAP:top=10': 0.48656349206349214, 'ERR:top=10': 0.7203265720605482, 'PrecisionAt:top=10': 0.6199999999999999, 'AverageGain:top=5': 0.5739130411297083, 'NDCG:top=5;type=Exp': 0.5484357889944593}}
    print(avg_of_scores(metrics_test_data, metrics_test_data))
