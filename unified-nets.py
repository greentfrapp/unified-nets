import tensorflow as tf
import numpy as np


def conv_dense():
	with tf.Session() as sess:
		input_ = tf.placeholder(
			shape=[None, 5, 1],
			dtype=tf.float32,
			name="input1",
		)
		input2_ = tf.placeholder(
			shape=[None, 5],
			dtype=tf.float32,
			name="input2",
		)
		conv = tf.layers.Conv1D(
			filters=5,
			kernel_size=5,
			activation=None,
			name="conv",
		)
		conv_output = conv.apply(input_)
		dense = tf.layers.Dense(
			units=5,
			activation=None,
			name="dense",
		)
		dense_output = dense.apply(input2_)
		dense_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in dense.trainable_weights]
		assigns = [tf.assign(v, p) for v, p in zip(dense.trainable_weights, dense_placeholders)]
		assign_op = tf.group(*assigns)

		sess.run(tf.global_variables_initializer())
		conv_weights = sess.run(conv.trainable_weights)
		dense_weights = []
		dense_weights.append(conv_weights[0].reshape(5,5))
		dense_weights.append(conv_weights[1])

		sess.run(assign_op, feed_dict=dict(zip(dense_placeholders, dense_weights)))

		rand_conv = np.random.randn(1, 5, 1)
		rand_dense = np.copy(rand_conv).reshape(-1, 5)
		conv_output, dense_output = sess.run([conv_output, dense_output], feed_dict={input_: rand_conv, input2_: rand_dense})

		# these should give the same values
		print(conv_output)
		print(dense_output)

def conv_rnn():
	with tf.Session() as sess:
		input_ = tf.placeholder(
			shape=[None, 5, 1],
			dtype=tf.float32,
			name="input1",
		)
		input2_ = tf.placeholder(
			shape=[None, 5, 1],
			dtype=tf.float32,
			name="input2",
		)
		conv = tf.layers.Conv1D(
			filters=5,
			kernel_size=2,
			activation=None,
			name="conv",
			padding="same",
		)
		conv_output = conv.apply(input_)

		rnn_cell = tf.nn.rnn_cell.BasicRNNCell(
			num_units=5,
			activation=None,
			# name="rnncell",
		)
		input_seq = [tf.squeeze(t, [1]) for t in tf.split(input2_, 5, 1)]
		rnn = tf.nn.static_rnn(
			cell=rnn_cell,
			inputs=input_seq,
			dtype=tf.float32,
		)

		conv_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in conv.trainable_weights]
		conv_assigns = [tf.assign(v, p) for v, p in zip(conv.trainable_weights, conv_placeholders)]
		conv_assign_op = tf.group(*conv_assigns)

		rnn_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in rnn_cell.trainable_weights]
		rnn_assigns = [tf.assign(v, p) for v, p in zip(rnn_cell.trainable_weights, rnn_placeholders)]
		rnn_assign_op = tf.group(*rnn_assigns)

		sess.run(tf.global_variables_initializer())
		
		# null first half of conv weights 
		# (might need to null 2nd half instead)
		conv_weights = sess.run(conv.trainable_weights)
		conv_weights[0][1, :, :] = np.zeros((1, 5))
		# print(conv_weights[0])
		sess.run(conv_assign_op, feed_dict=dict(zip(conv_placeholders, conv_weights)))

		rnn_weights = []
		rnn_weights.append(np.concatenate((conv_weights[0][0, :, :], np.zeros((5, 5))), axis=0))
		rnn_weights.append(conv_weights[1])
		# print(rnn_weights[0])
		# quit()
		sess.run(rnn_assign_op, feed_dict=dict(zip(rnn_placeholders, rnn_weights)))

		# null first 5 weights
		# (might need to null last 5 weights instead)
		# rnn_weights = sess.run(rnn_cell.trainable_weights)
		# rnn_weights[0][:5, :] = np.zeros((5, 5))

		rand_conv = np.random.randn(1, 5, 1)
		conv_output, rnn_output = sess.run([conv_output, rnn], feed_dict={input_: rand_conv, input2_: rand_conv})

		# these should give the same (similar) values
		print(conv_output)
		print(rnn_output)

def main():
	conv_rnn()


if __name__ == "__main__":
	main()