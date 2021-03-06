import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def mynet(input, reuse=False, embedding_size=100):
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, embedding_size, [1, 1], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)

        net = tf.contrib.layers.flatten(net)

    return net


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        # L2 distance between pairs
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keep_dims=True))

        matches_term = y * tf.square(d)

        # If the distance between non-matches is less then the margin, a positive term is added to
        # the loss (margin - d), thus the loss increases. If the distance is larger then the margin
        # (what we desire), the loss is not increased.
        nonmatches_term = (1 - y) * tf.square(tf.maximum((margin - d), 0))
        return tf.reduce_mean(matches_term + nonmatches_term) / 2
