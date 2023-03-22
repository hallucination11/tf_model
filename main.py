# 1.embedding_dimension = 5, batch_size = 256, lr = 0.01  auc = 0.84 实际正确率为0.75 no bn no drop
#
import numpy

import tensorflow as tf
import glob
import logging
import sys
import os
from features import int_features
from utils import batch_train_files
import time
from collections import defaultdict
import argparse
from model.CL_DSSM import CL_DSSM

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.disable_eager_execution()

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description="RUN MODEL !")
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--model', default='DNN')
    parser.add_argument('--sub_model', default=None)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--epoch', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--embedding_initializer', default=None)
    parser.add_argument('--dnn_weight_initializer', default=None)
    parser.add_argument('--dnn_weight_regularizer', default=None)
    parser.add_argument('--dropout', default=0.0)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--loss_type', default='sigmoid')
    parser.add_argument('--temperature', default=0.05)
    parser.add_argument('--task_type', default='binary')
    parser.add_argument('--cross_parameterization', default='vector')
    parser.add_argument('--cross_num', default=4)
    parser.add_argument('--gru_type', default="GRU")
    parser.add_argument('--use_neg', default=False)
    parser.add_argument('--alpha', default=1.0)
    parser.add_argument('--reduction_ratio', default=1.0)
    parser.add_argument('--bilinear_type', default='interaction')
    parser.add_argument('--num_groups', default=5)
    parser.add_argument('--bilinear_output_size', default=5)
    parser.add_argument('--vocabulary_size', default=100)
    parser.add_argument('--dat_emb_size', default=32)
    return parser.parse_args()


args = parse_args()

high_param = dict()
high_param['optimizer'] = args.optimizer
high_param['lr'] = args.lr
high_param['epoch'] = args.epoch
high_param['batch_size'] = args.batch_size
high_param['embedding_initializer'] = args.embedding_initializer
high_param['dnn_weight_initializer'] = args.dnn_weight_initializer
high_param['dnn_weight_regularizer'] = args.dnn_weight_regularizer
high_param['dropout'] = args.dropout
high_param['activation'] = args.activation
high_param['sub_model'] = args.sub_model
high_param['loss_type'] = args.loss_type
high_param['temperature'] = args.temperature
high_param['task_type'] = args.task_type
high_param['cross_parameterization'] = args.cross_parameterization
high_param['cross_num'] = args.cross_num
high_param['gru_type'] = args.gru_type
high_param['use_neg'] = args.use_neg
high_param['alpha'] = args.alpha
high_param['reduction_ratio'] = args.reduction_ratio
high_param['bilinear_type'] = args.bilinear_type
high_param['num_groups'] = args.num_groups
high_param['bilinear_output_size'] = args.bilinear_output_size
high_param['vocabulary_size'] = args.vocabulary_size
high_param['dat_emb_size'] = args.dat_emb_size


class ExampleHook(tf.compat.v1.train.SessionRunHook):
    def __init__(self):
        self.item = None
        self.uid_new = None
        self.id_list_embedding = None
        self.id_list_embedding_new = None
        self.sku_sn_embedding = None
        self.embedding_info = defaultdict()
        self.size = 0
        self.user_embedding = None

    def begin(self):
        # You can add ops to the graph here.
        # self.uid_embedding = tf.compat.v1.get_default_graph().get_tensor_by_name('input_layer/concat/concat:0')
        tf.compat.v1.logging.info('====Starting the session====')

    def after_create_session(self, session, coord):
        tf.compat.v1.logging.info('====Session created====')

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(
            [tf.compat.v1.train.get_global_step(), self.item])

    def after_run(self, run_context, run_values):
        global_step, labels = run_values.results
        # print(labels)

    def end(self, session):
        tf.compat.v1.logging.info('Done with the session.')


new_hook = ExampleHook()

model_dir = 'model_dir/'
saved_model_dir = 'saved_model_dir/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

if args.model == "CL_DSSM":
    model = CL_DSSM("CL_DSSM", model_dir=model_dir, embedding_upload_hook=new_hook, high_param=high_param)
logger.info("start training {} model".format(args.model))

feature_description = {}

# feature_description.update({
#     f: tf.io.VarLenFeature(tf.string)
#     for f in list_features
# })

feature_description.update({
    f: tf.io.FixedLenFeature([], tf.int64, 0)
    for f in int_features
})

feature_description['label'] = tf.io.FixedLenFeature([], tf.int64, 0)


def parser(record):
    read_data = tf.io.parse_example(serialized=record, features=feature_description)
    label = read_data.pop('label')

    return read_data, label


def get_input_fn(filename, batch_size, n_repeat):
    def input_fn():
        ds = tf.data.TFRecordDataset(filename)
        ds = ds.repeat(n_repeat).batch(batch_size)
        ds = ds.shuffle(buffer_size=batch_size)
        ds = ds.map(lambda x: parser(x))
        # ds = ds.prefetch(buffer_size=batch_size)
        return ds

    return input_fn


def main():
    start_time = time.time()
    training_files = []
    data_dir = 'D:/work/My_model_dataset/criteo/criteo_dataset/'
    training_files.extend(sorted(glob.glob(data_dir + 'output_file-part-*')))
    test_file = training_files.pop()
    logger.info("all training files size {}".format(len(training_files)))
    logger.info("test file is {}".format(test_file))
    # saved_model_dir = 'C:/Users/MSI/Desktop/code/data/data/saved_model'
    estimator = model.get_estimator()
    training_files_batched = batch_train_files(training_files, 4)
    # feature_spec = get_feature_spec(cnt_features, long_id_features, str_id_features, contains_features, list_features)
    for file_index, file_names in enumerate(training_files_batched):
        logger.info('Start training on files {}'.format(file_names))
        logger.info('training files size {}'.format(len(file_names)))
        start_time = time.time()
        estimator.train(
            input_fn=get_input_fn(
                filename=file_names,
                batch_size=32,
                n_repeat=1
            ),
            # hooks=[new_hook]
        )
        train_end_time = time.time()
        logger.info("training time {} s".format(train_end_time - start_time))

        train_metrics = estimator.evaluate(
            input_fn=get_input_fn(
                filename=file_names[:1],
                batch_size=32,
                n_repeat=1
            ),
            name='train_eval'
        )

        evaluate_end_time = time.time()
        logger.info("evaluate time is {}".format(evaluate_end_time - train_end_time))
        logger.info("train_metrics is {}".format(train_metrics))

        test_metrics = estimator.evaluate(
            input_fn=get_input_fn(
                filename=test_file,
                batch_size=32,
                n_repeat=1
            ),
            name='test_result'
        )
        test_end_time = time.time()
        logger.info("test metrics is \n{}".format(test_metrics))

    export_path = estimator.export_saved_model(saved_model_dir,
                                               tf.estimator.export.build_parsing_serving_input_receiver_fn(
                                                   feature_spec=feature_description))
    end_time = time.time()
    logger.info("time need {}".format(end_time - start_time))


if __name__ == '__main__':
    main()
