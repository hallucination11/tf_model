import numpy as np
import tensorflow as tf
import collections
from tensorflow.python.keras.layers import Layer, Dropout
from collections import Counter
from tensorflow.python.keras.layers import Dense, Lambda

from tensorflow.python.keras.layers import (Concatenate, Dense, Permute, multiply, Flatten)
from tensorflow.python.keras.layers import Embedding


class Model(collections.namedtuple("Model", ["model_name",
                                             'model_dir', 'embedding_upload_hook', 'high_param'])):
    def __new__(cls,
                model_name,
                model_dir,
                embedding_upload_hook=None,
                high_param=None
                ):
        return super(Model, cls).__new__(
            cls,
            model_name,
            model_dir,
            embedding_upload_hook,
            high_param
        )

    def get_model_fn(self):
        def model_fn(features, labels, mode, params):
            pass

        return model_fn

    def get_estimator(self):
        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={}
        )

        # add gauc

        return estimator