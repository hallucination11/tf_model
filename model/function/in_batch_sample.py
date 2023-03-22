import tensorflow as tf
import numpy as np

batch_size = 4
user_list = [0.12, 0.34, 0.21, 0.43]
user_list = np.reshape(user_list, [4, 1])
item_list = [0.55, 0.12, 0.32, 0.22]
item_list = np.reshape(item_list, [4, 1])

item_list_new = item_list

NEG = 2

for i in range(NEG):
    ss = tf.gather(item_list, tf.random.shuffle(tf.range(tf.shape(item_list)[0])))
    item_list_new = tf.concat([item_list_new, ss], axis=0)

prod_raw = tf.reduce_sum(tf.multiply(tf.tile(user_list, [NEG + 1, 1]), item_list_new), 1, True)
prod = tf.transpose(tf.reshape(tf.transpose(prod_raw), [NEG + 1, batch_size]))
print(prod)
a = tf.nn.softmax(prod)
print(a)
# 只取第一列，即正样本列概率。
hit_prob = tf.slice(a, [0, 0], [-1, 1])
print(hit_prob)
loss = -tf.reduce_mean(tf.compat.v1.log(hit_prob))

print(tf.nn.softmax(
    [[0.066, 0.066, 0.0264],
     [0.0408, 0.0408, 0.0408],
     [0.0672, 0.0462, 0.1155],
     [0.0946, 0.1376, 0.1376]]))
