from model.parent_model import Model
import tensorflow as tf


class FRNet(Model):
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