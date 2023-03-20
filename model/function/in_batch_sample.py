import tensorflow as tf
import numpy as np

batch_size = 4
user_list = [1, 2, 3, 4]
user_list = np.reshape(user_list, [4, 1])
item_list = [5, 6, 7, 8]
item_list = np.reshape(item_list, [4, 1])

item_list_new = item_list

NEG = 2

for i in range(NEG):
    ss = tf.gather(item_list, tf.random.shuffle(tf.range(tf.shape(item_list)[0])))
    item_list_new = tf.concat([item_list_new, ss], axis=0)

prod_raw = tf.reduce_sum(tf.multiply(tf.tile(user_list, [NEG + 1, 1]), item_list_new), 1, True)
prod = tf.transpose(tf.reshape(tf.transpose(prod_raw), [NEG + 1, batch_size]))
print(prod)