import tensorflow as tf
import numpy as np

batch_size = 4
user_list = [1.0, 2.0, 3.0, 4.0]
user_list = np.reshape(user_list, [4, 1])
item_list = [5.0, 6.0, 7.0, 8.0]
item_list = np.reshape(item_list, [4, 1])

item_list_new = item_list

NEG = 2

for i in range(NEG):
    ss = tf.gather(item_list, tf.random.shuffle(tf.range(tf.shape(item_list)[0])))
    item_list_new = tf.concat([item_list_new, ss], axis=0)

prod_raw = tf.reduce_sum(tf.multiply(tf.tile(user_list, [NEG + 1, 1]), item_list_new), 1, True)
prod = tf.transpose(tf.reshape(tf.transpose(prod_raw), [NEG + 1, batch_size]))
print(prod)
prob = tf.nn.softmax(prod)
print(prod)
# 只取第一列，即正样本列概率。
hit_prob = tf.slice(prob, [0, 0], [-1, 1])
print(hit_prob)
loss = -tf.reduce_mean(tf.compat.v1.log(hit_prob))
print(loss)