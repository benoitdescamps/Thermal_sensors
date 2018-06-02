import tensorflow as tf

def define_Q(input_shape=(16,16)):
    input = tf.placeholder(shape=(None,)+input_shape+(1,), dtype=tf.float32)
    nn_1 = tf.layers.batch_normalization(input)
    filter_1 = tf.Variable(tf.random_normal([3, 3, 1, 4], stddev=1.0))
    layer_1 = tf.nn.conv2d(input=nn_1, strides=[1, 1, 1, 1], filter=filter_1, padding='VALID')
    filter_2 = tf.Variable(tf.random_normal([3, 3, 4, 8], stddev=1.0))
    layer_2 = tf.nn.conv2d(input=layer_1, strides=[1, 1, 1, 1], filter=filter_2, padding='VALID')
    filter_3 = tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=1.0))
    layer_3 = tf.nn.conv2d(input=layer_2, strides=[1, 1, 1, 1], filter=filter_3, padding='VALID')
    #layer_2 = tf.layers.max_pooling2d(inputs=layer_1, pool_size=(4, 4), strides=(1, 1))
    #remove maxpooling layer as translational invariance might not be necessary here
    layer_3 = tf.layers.dense(inputs=tf.layers.Flatten()(layer_3) \
                              , units=8, activation=tf.nn.relu, use_bias=True)
    output = tf.layers.dense(inputs=layer_3, units=3, use_bias=True)
    return input,output

def get_cost(target,Q,action_indices):
    row_indices =  tf.range(tf.shape(action_indices)[0])
    full_indices = tf.stack([row_indices, action_indices], axis=1)
    q_values = tf.gather_nd(Q, full_indices)
    return tf.losses.mean_squared_error(labels=target,predictions=q_values)
