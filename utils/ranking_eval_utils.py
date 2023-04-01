from copy import deepcopy


import numpy as np
import pandas as pd
from catboost import Pool
from catboost.utils import eval_metric

METRICS = ['NDCG', 'NDCG:top=10', 'NDCG:top=5', 'ERR', 'ERR:top=10', 'ERR:top=5',
               'PrecisionAt:top=10', 'PrecisionAt:top=5', 'AverageGain:top=10', 'AverageGain:top=5',
               'MAP:top=10', 'MAP:top=5']
def get_metrics(skip_train=False):
    metrics_to_return = deepcopy(METRICS)
    if not skip_train:
        for idx, metric in enumerate(metrics_to_return):
            param_delimiter = ':'
            if metric.__contains__(':'):
                param_delimiter = ';'
            metrics_to_return[idx] = metric + param_delimiter + 'hints=skip_train~false'
    return metrics_to_return
def eval_predictions(predictions, group_ids, metrics=METRICS):
    predictions = predictions.to_numpy()
    group_ids = group_ids.to_numpy()
    df = pd.DataFrame({'prediction':predictions, 'group_id':group_ids})
    df_grouped = df.groupby('group_id')
    df['prediction'] = df_grouped['prediction'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
    df = df.sort_values('group_id', ascending=False).reset_index()
    df['sorted_prediction'] = (df.sort_values(by=['group_id', 'prediction'], ascending=False)).reset_index()['prediction']
    metrics_output = {}
    for metric in metrics:
         metrics_output[metric] = \
            eval_metric(label=df['sorted_prediction'].to_numpy(), approx=df['prediction'], group_id=df['group_id'],
                        metric=metric)[0]
    return metrics_output
    # for metric in metrics:


if __name__ == '__main__':
    MAX_RANK = 46
    Q_NUMBER = 20
    ranks = np.linspace(1, MAX_RANK, MAX_RANK)
    pred_df = pd.DataFrame(columns=['pred', 'qid'])
    for qid in range(1, Q_NUMBER + 1):
        group = pd.DataFrame({'pred': ranks})
        group.insert(column='qid', value=qid, loc=1)
        pred_df = pd.concat([pred_df, group], ignore_index=True)
    pred_df = pred_df.groupby('qid').sample(frac=1)
    eval_predictions(pred_df['pred'], pred_df['qid'])
    print(get_metrics())


    #
    # def fit_model_random(test_pool):
    #     eval_preds()


