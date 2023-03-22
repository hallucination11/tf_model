from model.parent_model import Model
import tensorflow as tf
import random
from tensorflow.python.keras.layers import Lambda


class CL_DSSM(Model):
    def get_model_fn(self):

        def _cosine(query_encoder, doc_encoder, params):
            NEG = params['NEG']
            doc_encoder_fd = doc_encoder
            for i in range(NEG):
                ss = tf.gather(doc_encoder, tf.random.shuffle(tf.range(tf.shape(doc_encoder)[0])))
                doc_encoder_fd = tf.concat([doc_encoder_fd, ss], axis=0)
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_encoder), axis=1, keepdims=True)),
                                 [NEG + 1, 1])
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_encoder_fd), axis=1, keepdims=True))
            query_encoder_fd = tf.tile(query_encoder, [NEG + 1, 1])
            prod = tf.reduce_sum(tf.multiply(query_encoder_fd, doc_encoder_fd, name="sim-multiply"), axis=1,
                                 keepdims=True)
            norm_prod = tf.multiply(query_norm, doc_norm)
            cos_sim_raw = tf.truediv(prod, norm_prod)
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, -1])) * 20

            prob = tf.nn.softmax(cos_sim, name="sim-softmax")
            hit_prob = tf.slice(prob, [0, 0], [-1, 1], name="sim-slice")
            loss = -tf.reduce_mean(tf.compat.v1.log(hit_prob), name="sim-mean")
            return loss

        def model_fn(features, labels, mode, params):
            user_feature_embeddings = []
            item_feature_embeddings = []
            feature_square_embeddings = []
            feature_embeddings = []
            self.embedding_upload_hook.item = labels
            Utower_features = ['uid', 'gender', 'bal', 'mobile_level', 'age']
            Itower_features = ['item', 'brand_id', 'prod_id', 'item_price']

            # utower
            for feature in Utower_features:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                user_feature_embeddings.append(feature_emb)

            uTower_input = tf.concat(user_feature_embeddings, axis=1, name='utower')

            for unit in params['hidden_units']:
                uTower_output = tf.compat.v1.layers.dense(uTower_input, units=unit, activation=tf.nn.relu)
                uTower_output = tf.compat.v1.layers.batch_normalization(uTower_output)
                uTower_output = tf.compat.v1.layers.dropout(uTower_output)
                uTower_output = tf.nn.l2_normalize(uTower_output)

            # itower
            for feature in Itower_features:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                item_feature_embeddings.append(feature_emb)

            item_feature_embedding_1 = random.sample(item_feature_embeddings, 3)
            item_feature_embedding_2 = random.sample(item_feature_embeddings, 3)

            iTower_input_origin = tf.concat(item_feature_embeddings, axis=1, name='itower_origin')
            iTower_input_1 = tf.concat(item_feature_embedding_1, axis=1, name='itower_1')
            iTower_input_2 = tf.concat(item_feature_embedding_2, axis=1, name='itower_2')

            for unit in params['tower_units']:
                iTower_output_origin = tf.compat.v1.layers.dense(iTower_input_origin, units=unit, activation=tf.nn.relu)
                iTower_output_origin = tf.compat.v1.layers.batch_normalization(iTower_output_origin)
                iTower_output_origin = tf.compat.v1.layers.dropout(iTower_output_origin)
                iTower_output_origin = tf.nn.l2_normalize(iTower_output_origin)

                iTower_output_1 = tf.compat.v1.layers.dense(iTower_input_1, units=unit, activation=tf.nn.relu)
                iTower_output_1 = tf.compat.v1.layers.batch_normalization(iTower_output_1)
                iTower_output_1 = tf.compat.v1.layers.dropout(iTower_output_1)
                iTower_output_1 = tf.nn.l2_normalize(iTower_output_1)

                iTower_output_2 = tf.compat.v1.layers.dense(iTower_input_2, units=unit, activation=tf.nn.relu)
                iTower_output_2 = tf.compat.v1.layers.batch_normalization(iTower_output_2)
                iTower_output_2 = tf.compat.v1.layers.dropout(iTower_output_2)
                iTower_output_2 = tf.nn.l2_normalize(iTower_output_2)

            # maximize
            maximum = tf.multiply(iTower_output_1, tf.compat.v1.random_shuffle(iTower_output_1) / params['temperature'])

            # minimize
            minimum = tf.multiply(iTower_output_1, iTower_output_2) / params['temperature']
            cl_logit = -tf.reduce_mean(tf.compat.v1.log(tf.exp(maximum) / tf.exp(minimum)))
            print(cl_logit)
            exit()

            # crossTower
            # 此处为user和item的交叉特征和统计特征，这里用item和uid交叉
            sum_embedding_then_square = tf.square(tf.add_n(feature_embeddings))
            square_embedding_then_sum = tf.add_n(feature_square_embeddings)
            fm_output = 0.5 * (sum_embedding_then_square - square_embedding_then_sum)
            fm_output = tf.compat.v1.layers.dense(fm_output, units=params['tower_units'][-1])

            final_input = tf.concat(uTower_output + iTower_output + fm_output, axis=1, name='final_logits')

            tf.compat.v1.logging.info("output shape={}".format(final_input.shape))

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

        def get_bucket_column(key, boundary, dimension):
            num_col = tf.feature_column.numeric_column(key)
            bucket_column = tf.feature_column.bucketized_column(num_col, boundaries=boundary)
            return tf.feature_column.embedding_column(bucket_column, dimension=dimension)

        cnt_feature_columns = {
            "uid": get_categorical_hash_bucket_column("uid", hash_bucket_size=2000, dimension=4, dtype=tf.int64),
            "item": get_categorical_hash_bucket_column("item", hash_bucket_size=100, dimension=4, dtype=tf.int64),
            "bal": get_bucketized_column("bal", boundaries=[10002.0, 14158.35, 18489.0, 23177.0, 27839.8, 32521.5,
                                                            36666.7, 41386.9, 45919.6, 50264.55, 54345.0], dimension=4),
            "gender": get_categorical_hash_bucket_column("gender", hash_bucket_size=4, dimension=1, dtype=tf.int64),
            "brand_id": get_categorical_hash_bucket_column("brand_id", hash_bucket_size=1000, dimension=6,
                                                           dtype=tf.int64),
            "prod_id": get_categorical_hash_bucket_column("prod_id", hash_bucket_size=1000, dimension=3,
                                                          dtype=tf.int64),
            "age": get_categorical_hash_bucket_column("age", hash_bucket_size=1000, dimension=3, dtype=tf.int64),
            "mobile_level": get_categorical_hash_bucket_column("mobile_level", hash_bucket_size=4, dimension=2,
                                                               dtype=tf.int64),
            "item_price": get_bucket_column("item_price",
                                            boundary=[0.0200, 47.4695, 98.1180, 148.7500, 201.6680, 258.0025, 306.8860,
                                                      354.4485, 404.2520, 456.1920, 508.1800, 555.3115, 604.0460,
                                                      654.5650, 703.6780, 754.3950, 807.0580, 855.0515, 902.4010,
                                                      951.3025], dimension=4)

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
                'temperature': self.high_param['temperature']
            })

        return estimator
