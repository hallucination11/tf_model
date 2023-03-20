import tensorflow as tf
import csv
from tqdm import *


def build_int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))


def build_int64list_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=data))


# Generate Float Features.
def build_float_feature(data):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))


# Generate String Features.
def build_string_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(data).encode()]))


def convert_to_tfexample(application, perchase, uid, gender, bal, prod_list, item, seq_len, label, brand_id, prod_id,
                         age, mobile_level, item_price):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'application': build_int64_feature(application),
                'approval': build_int64_feature(perchase),
                'uid': build_int64_feature(uid),
                'gender': build_int64_feature(gender),
                'bal': build_int64_feature(bal),
                'prod_list': build_int64list_feature(prod_list),
                'item': build_int64_feature(item),
                'seq_len': build_int64_feature(seq_len),
                'label': build_int64_feature(label),
                'brand_id': build_int64_feature(brand_id),
                'prod_id': build_int64_feature(prod_id),
                'age': build_int64_feature(age),
                'mobile_level': build_int64_feature(mobile_level),
                'item_price': build_float_feature(item_price)
            })
    )


def write_tf_records():
    writer = tf.io.TFRecordWriter('dataset/part-r-1.tfrecords')
    with open('random_dataset.csv', encoding='gbk') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for i, record in tqdm(enumerate(reader)):
            if i == 0:
                continue
            application, perchase, uid, gender, bal, prod_list, item, seq_len, label, brand_id, prod_id, age, \
            mobile_level, item_price = record[0:14]

            prod_arr = list()
            for _ in prod_list.split(','):
                prod_arr.append(int(_))
            example = convert_to_tfexample(int(application), int(perchase), int(uid), int(gender), int(bal), prod_arr,
                                           int(item), int(seq_len), int(label), int(brand_id), int(prod_id), int(age),
                                           int(mobile_level), float(item_price))
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    write_tf_records()
