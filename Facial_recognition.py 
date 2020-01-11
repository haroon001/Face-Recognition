import tensorflow as tf

class face_recognition:
	def __init__():
	

		self.image_size = 40
		self.no_landmark = 10
		self.no_gender_classes = 2
		self.no_smile_classes = 2
		self.no_glasses_classes = 2
		self.no_headpose_classes = 5
		self.batch_size = 100
		self.total_batches = 300

		self.image_input = tf.placeholder(tf.float32, shape=[None, image_size, image_size])
		self.landmark_input = tf.placeholder(tf.float32, shape=[None, no_landmark])
		self.gender_input = tf.placeholder(tf.float32, shape=[None, no_gender_classes])
		self.smile_input = tf.placeholder(tf.float32, shape=[None, no_smile_classes])
		self.glasses_input = tf.placeholder(tf.float32, shape=[None, no_glasses_classes])
		self.headpose_input = tf.placeholder(tf.float32, shape=[None, no_headpose_classes])

	def Network():

		image_input_reshape = tf.reshape(self.image_input, [-1, image_size, image_size, 1],name='input_reshape')

		convolution_layer_1 = convolution_layer(image_input_reshape, 16)
		pooling_layer_1 = pooling_layer(convolution_layer_1)
		convolution_layer_2 = convolution_layer(pooling_layer_1, 48)
		pooling_layer_2 = pooling_layer(convolution_layer_2)
		convolution_layer_3 = convolution_layer(pooling_layer_2, 64)
		pooling_layer_3 = pooling_layer(convolution_layer_3)
		convolution_layer_4 = convolution_layer(pooling_layer_3, 64)
		flattened_pool = tf.reshape(convolution_layer_4, [-1, 5 * 5 * 64],name='flattened_pool')
		dense_layer_bottleneck = dense_layer(flattened_pool, 1024)
		dropout_bool = tf.placeholder(tf.bool)

		dropout_layer = tf.layers.dropout(inputs=dense_layer_bottleneck,rate=0.4,training=dropout_bool)


		landmark_logits = dense_layer(dropout_layer, 10)
		smile_logits = dense_layer(dropout_layer, 2)
		glass_logits = dense_layer(dropout_layer, 2)
		gender_logits = dense_layer(dropout_layer, 2)
		headpose_logits = dense_layer(dropout_layer, 5)

		landmark_loss = 0.5 * tf.reduce_mean(tf.square(landmark_input, landmark_logits))
		gender_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gender_input, logits=gender_logits))

		smile_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=smile_input, logits=smile_logits))

		glass_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=glasses_input, logits=glass_logits))

		headpose_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=headpose_input, logits=headpose_logits))

		loss_operation = landmark_loss + gender_loss + smile_loss + glass_loss + headpose_loss

		optimiser = tf.train.AdamOptimizer().minimize(loss_operation)

		return loss_operation