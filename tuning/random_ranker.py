import re
from copy import deepcopy

import tensorboard
import tensorflow as tf

import pandas as pd
from catboost import Pool

from utils.ranking_eval_utils import eval_predictions


class ShuffleRanker:
    def __init__(self,
                 custom_metric: list,
                 iterations=None,
                 learning_rate=None,
                 depth=None,
                 l2_leaf_reg=None,
                 model_size_reg=None,
                 rsm=None,
                 loss_function='YetiRank',
                 border_count=None,
                 feature_border_type=None,
                 per_float_feature_quantization=None,
                 input_borders=None,
                 output_borders=None,
                 fold_permutation_block=None,
                 od_pval=None,
                 od_wait=None,
                 od_type=None,
                 nan_mode=None,
                 counter_calc_method=None,
                 leaf_estimation_iterations=None,
                 leaf_estimation_method=None,
                 thread_count=None,
                 random_seed=None,
                 use_best_model=None,
                 best_model_min_trees=None,
                 verbose=None,
                 silent=None,
                 logging_level=None,
                 metric_period=None,
                 ctr_leaf_count_limit=None,
                 store_all_simple_ctr=None,
                 max_ctr_complexity=None,
                 has_time=None,
                 allow_const_label=None,
                 target_border=None,
                 one_hot_max_size=None,
                 random_strength=None,
                 name=None,
                 ignored_features=None,
                 train_dir=None,
                 eval_metric=None,
                 bagging_temperature=None,
                 save_snapshot=None,
                 snapshot_file=None,
                 snapshot_interval=None,
                 fold_len_multiplier=None,
                 used_ram_limit=None,
                 gpu_ram_part=None,
                 pinned_memory_size=None,
                 allow_writing_files=None,
                 final_ctr_computation_mode=None,
                 approx_on_full_history=None,
                 boosting_type=None,
                 simple_ctr=None,
                 combinations_ctr=None,
                 per_feature_ctr=None,
                 ctr_description=None,
                 ctr_target_border_count=None,
                 task_type=None,
                 device_config=None,
                 devices=None,
                 bootstrap_type=None,
                 subsample=None,
                 mvs_reg=None,
                 sampling_frequency=None,
                 sampling_unit=None,
                 dev_score_calc_obj_block_size=None,
                 dev_efb_max_buckets=None,
                 sparse_features_conflict_fraction=None,
                 max_depth=None,
                 n_estimators=None,
                 num_boost_round=None,
                 num_trees=None,
                 colsample_bylevel=None,
                 random_state=None,
                 reg_lambda=None,
                 objective=None,
                 eta=None,
                 max_bin=None,
                 gpu_cat_features_storage=None,
                 data_partition=None,
                 metadata=None,
                 early_stopping_rounds=None,
                 cat_features=None,
                 grow_policy=None,
                 min_data_in_leaf=None,
                 min_child_samples=None,
                 max_leaves=None,
                 num_leaves=None,
                 score_function=None,
                 leaf_estimation_backtracking=None,
                 ctr_history_unit=None,
                 monotone_constraints=None,
                 feature_weights=None,
                 penalties_coefficient=None,
                 first_feature_use_penalties=None,
                 per_object_feature_penalties=None,
                 model_shrink_rate=None,
                 model_shrink_mode=None,
                 langevin=None,
                 diffusion_temperature=None,
                 posterior_sampling=None,
                 boost_from_average=None,
                 text_features=None,
                 tokenizers=None,
                 dictionaries=None,
                 feature_calcers=None,
                 text_processing=None,
                 embedding_features=None,
                 eval_fraction=None,
                 *args,
                 **kwargs):
        self.custom_metric = self.__remove_hint_skip_train_from_list(custom_metric)
        self.iterations = iterations
        self.train_dir = train_dir
        pass
    def __remove_hint_skip_train_from_list(self, str_list: list):
        pattern = r";hints"
        str_list = [re.split(pattern, entry)[0] for entry in str_list]
        return str_list
    def fit(self, X: Pool, eval_set: Pool, plot, *args, **kwargs):
        # train
        self.train_metrics = self.__metrics_for_random_ranking(X)
        self.__fake_fitting_with_tf_writer(self.train_metrics, 'learn')

        # test
        self.test_metrics = self.__metrics_for_random_ranking(eval_set)
        self.__fake_fitting_with_tf_writer(self.test_metrics, 'test')

    def __fake_fitting_with_tf_writer(self, metric_values: dict, set_name: str):
        summary_writer = tf.summary.create_file_writer(self.train_dir + '/' + set_name)
        with summary_writer.as_default():
            for i in range(1, self.iterations + 1):
                for metric in self.custom_metric:
                    tf.summary.scalar(metric, metric_values[metric], step=i)

    def __metrics_for_random_ranking(self, pool: Pool):
        data = pool
        preds = pd.DataFrame(columns=['pred', 'qid'], data={'pred': data.get_label(), 'qid' : data.get_group_id_hash()})
        preds = preds.groupby('qid').sample(frac=1)
        return eval_predictions(preds['pred'], preds['qid'], metrics=self.custom_metric)

    @property
    def best_score_(self):
        return self.get_best_score()

    def get_best_score(self):
        print({'train_set:': self.train_metrics, 'test_set': self.test_metrics}.__str__())


default_parameters = {
    'iterations': 100,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0
}


def fit_model(loss_function, train_pool, test_pool, additional_params=None):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = ShuffleRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model


if __name__ == '__main__':
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

    parameters = {}
    model = fit_model('YetiRankPairwise',
                      additional_params=
                      {'custom_metric': ['NDCG:top=5;type=Exp;hints_skip_train', 'PrecisionAt:top=10', 'RecallAt:top=10', 'MAP:top=10'], 'train_dir' : 'ShuffleRanker'},
                      train_pool=train,
                      test_pool=test)
    model.best_score_
