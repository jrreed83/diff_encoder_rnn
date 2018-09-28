import tensorflow as tf 
import numpy as np 

batch_size = 2
seq_len = 4
feat_size = 2
hidden_size = 4 

x = tf.placeholder(name='x', dtype=tf.float32, shape=(batch_size, seq_len, feat_size))
t = tf.placeholder(name='t', dtype=tf.int32, shape=(batch_size, seq_len))

# Define the model
gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
initial_state = gru_cell.zero_state(batch_size, dtype=tf.float32)
outputs, _ = tf.nn.dynamic_rnn(
    gru_cell, 
    x, 
    dtype=tf.float32, 
    initial_state = initial_state
)

# Flatten out
outputs = tf.reshape(outputs, [2, -1, hidden_size])
targets = tf.reshape(t, [2, -1])
outputs = tf.layers.dense(outputs, 2)

# Execute the model
loss = tf.losses.sparse_softmax_cross_entropy(
    labels=targets, logits=outputs)

init = tf.global_variables_initializer()

# Perform the training
with tf.Session() as sess:
    sess.run(init)
    x_ = np.random.randn(batch_size, seq_len, feat_size)
    t_ = np.array([[1,0,1,0],[0,1,0,1]])
    o = sess.run(outputs, feed_dict={x: x_})
    y_ = sess.run(loss, feed_dict={x: x_, t: t_})
    t_ = sess.run(targets, feed_dict={t:t_})
    print(y_)
    print(o)
    print(t_)