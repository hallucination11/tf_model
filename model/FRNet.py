from model.parent_model import Model
import tensorflow as tf


class FRNet(Model):
    def get_model_fn(self):

        def model_fn(features, labels, mode, params):
            user_feature_embeddings = []
            item_feature_embeddings = []
            feature_square_embeddings = []
            feature_embeddings = []
            self.embedding_upload_hook.item = labels
            Utower_features = ['uid', 'gender', 'bal']
            Itower_features = ['item']
            CrossTower_features = ['uid', 'item']

            # utower
            for feature in Utower_features:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                user_feature_embeddings.append(feature_emb)
                if feature == 'uid':
                    cross_emb_1 = feature_emb
                    feature_square_embeddings.append(tf.square(feature_emb))
                    feature_embeddings.append(cross_emb_1)

            uTower_input = tf.concat(user_feature_embeddings, axis=1, name='utower')

            for unit in params['hidden_units']:
                uTower_output = tf.compat.v1.layers.dense(uTower_input, units=unit, activation=tf.nn.relu)
                uTower_output = tf.compat.v1.layers.batch_normalization(uTower_output)
                uTower_output = tf.compat.v1.layers.dropout(uTower_output)

            # itower
            for feature in Itower_features:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                item_feature_embeddings.append(feature_emb)
                if feature == 'item':
                    feature_square_embeddings.append(tf.square(feature_emb))
                    feature_embeddings.append(feature_emb)

            iTower_input = tf.concat(item_feature_embeddings, axis=1, name='itower')

            for unit in params['tower_units']:
                iTower_output = tf.compat.v1.layers.dense(iTower_input, units=unit, activation=tf.nn.relu)
                iTower_output = tf.compat.v1.layers.batch_normalization(iTower_output)
                iTower_output = tf.compat.v1.layers.dropout(iTower_output)

            # crossTower
            # 此处为user和item的交叉特征和统计特征，这里用item和uid交叉
            sum_embedding_then_square = tf.square(tf.add_n(feature_embeddings))
            square_embedding_then_sum = tf.add_n(feature_square_embeddings)
            fm_output = 0.5 * (sum_embedding_then_square - square_embedding_then_sum)
            fm_output = tf.compat.v1.layers.dense(fm_output, units=params['tower_units'][-1])

            final_input = tf.concat(uTower_output + iTower_output + fm_output, axis=1, name='final_logits')

            tf.compat.v1.logging.info("output shape={}".format(final_input.shape))

            for unit in params['hidden_units']:
                application_output = tf.compat.v1.layers.dense(final_input, units=unit, activation=tf.nn.relu)
                application_output = tf.compat.v1.layers.batch_normalization(application_output)
                application_output = tf.compat.v1.layers.dropout(application_output)

            application_logit = tf.compat.v1.layers.dense(application_output, units=1, activation=tf.nn.relu)

            for unit in params['hidden_units']:
                approval_output = tf.compat.v1.layers.dense(final_input, units=unit, activation=tf.nn.relu)
                approval_output = tf.compat.v1.layers.batch_normalization(approval_output)
                approval_output = tf.compat.v1.layers.dropout(approval_output)

            approval_logit = tf.compat.v1.layers.dense(approval_output, units=1, activation=tf.nn.relu)

            # 应该先写predict，因为mode位predict或infer时，labels默认为none
            if mode == tf.estimator.ModeKeys.PREDICT:
                application_pred = tf.sigmoid(application_logit, name='application_pred')
                approval_pred = tf.sigmoid(approval_logit, name='approval_pred')

                predictions = {
                    'application_probabilities': application_pred,
                    'approval_probabilities': approval_pred
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            # compute loss
            application_pred = tf.sigmoid(application_logit, name='application_pred')
            application_loss = tf.compat.v1.losses.log_loss(labels, application_pred)

            approval_pred = tf.sigmoid(approval_logit, name='approval_pred')
            approval_loss = tf.compat.v1.losses.log_loss(labels, approval_pred)

            loss = application_loss + approval_loss

            if mode == tf.estimator.ModeKeys.TRAIN:
                # optimizer
                if self.high_param['optimizer'] == 'Adam':
                    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
                if self.high_param['optimizer'] == 'Adgrad':
                    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.005)
                if self.high_param['optimizer'] == 'PAO':
                    optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=0.001)
                train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            if mode == tf.estimator.ModeKeys.EVAL:
                application_pred = tf.sigmoid(application_logit, name='click_pred')
                application_pred = tf.argmax(application_pred, axis=-1)
                application_auc = tf.compat.v1.metrics.auc(labels=labels, predictions=application_pred)

                approval_pred = tf.sigmoid(approval_logit, name='click_pred')
                approval_pred = tf.argmax(approval_pred, axis=-1)
                approval_auc = tf.compat.v1.metrics.auc(labels=labels, predictions=approval_pred)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'approval_auc': approval_auc,
                                                                                    'application_auc': application_auc})

        return model_fn

    def get_estimator(self):
        # 商品id类特征
        def get_categorical_hash_bucket_column(key, hash_bucket_size, dimension, dtype):
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key, hash_bucket_size=hash_bucket_size, dtype=dtype
            )
            return tf.feature_column.embedding_column(categorical_column, dimension=dimension)

        # 连续值类特征（差异较为明显）
        def get_bucketized_column(key, boundaries, dimension):
            bucketized_column = tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(key), boundaries)
            return tf.feature_column.embedding_column(bucketized_column, dimension=dimension)

        # 层级分类类型特征(num_bucket需要按实际赋值)
        def get_categorical_identity_column(key, num_buckets, dimension, default_value=0):
            identity_column = tf.feature_column.categorical_column_with_identity(key, num_buckets=num_buckets,
                                                                                 default_value=default_value)
            return tf.feature_column.embedding_column(identity_column, dimension=dimension)

        cnt_feature_columns = {
            "uid": get_categorical_hash_bucket_column("uid", hash_bucket_size=2000, dimension=4, dtype=tf.int64),
            "item": get_categorical_hash_bucket_column("item", hash_bucket_size=100, dimension=4, dtype=tf.int64),
            "bal": get_bucketized_column("bal", boundaries=[10002.0, 14158.35, 18489.0, 23177.0, 27839.8, 32521.5,
                                                            36666.7, 41386.9, 45919.6, 50264.55, 54345.0], dimension=4),
            "gender": get_categorical_hash_bucket_column("gender", hash_bucket_size=4, dimension=1, dtype=tf.int64)
        }

        all_feature_column = {}
        all_feature_column.update(cnt_feature_columns)

        # weight_column = tf.feature_column.numeric_column('weight')

        hidden_layers = [256, 128]

        tower_layers = [512, 256, 128]

        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={
                'hidden_units': hidden_layers,
                'tower_units': tower_layers,
                'feature_columns': all_feature_column,
                # 'weight_column': weight_column,
            })

        return estimator