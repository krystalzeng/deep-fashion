from code.net_model import *
from code.pretrained_net_model import *
from code.image_dataset import BatchGenerator, get_fashion_dataset
from tensorflow.contrib.slim.nets import vgg
import tensorflow.contrib.slim as slim
import os

flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('train_iter', 10, 'Total training iter')
flags.DEFINE_integer('step', 5, 'Save after ... iteration')
flags.DEFINE_bool('is_vgg', False, 'Whether to use the pretrained vgg-16 or the custom cnn')

###############################################
# 1. Create the data generator
###############################################
dataset, image_dict = get_fashion_dataset('list_dataset.pickle', '../full/')
gen = BatchGenerator(dataset, image_dict, testset_proportion=0.1)

##################################################################
# 2. Define the training procedure
##################################################################
left = tf.placeholder(tf.float32, [None, 150, 150, 3], name='left')
right = tf.placeholder(tf.float32, [None, 150, 150, 3], name='right')

# required for vgg
resized_left = tf.image.resize_images(left, [224, 224])
resized_right = tf.image.resize_images(right, [224, 224])

with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
    label = tf.to_float(label)
margin = 0.9

left_output = mynet(left, reuse=False)
right_output = mynet(right, reuse=True)

with slim.arg_scope(vgg.vgg_arg_scope()):
    left_output_vgg = vgg_16(resized_left, num_classes=100, is_training=True, reuse=False)
    right_output_vgg = vgg_16(resized_right, num_classes=100, is_training=True, reuse=True)

global_step = tf.Variable(0, trainable=False)

if FLAGS.is_vgg:
    loss = contrastive_loss(left_output_vgg, right_output_vgg, margin)
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join('vgg-16/', 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
else:
    loss = contrastive_loss(left_output, right_output, label, margin)

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

saver = tf.train.Saver()

######################################################################
# 3. Run the model
######################################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # setup tensor board
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('train.log', sess.graph)

    # train iter
    for i in range(FLAGS.train_iter):
        b_l, b_r, b_label = gen.next_batch(FLAGS.batch_size, 0.5)

        _, l, summary_str = sess.run([train_step, loss, merged],
                                     feed_dict={left: b_l, right: b_r, label: b_label})

        writer.add_summary(summary_str, i)
        print("\r#%d - Loss" % i, l)

        if (i + 1) % FLAGS.step == 0:
            t_l, t_r, t_label = gen.test_set()
            # generate test
            l = sess.run(loss, feed_dict={left: t_l, right: t_r, label: t_label})
            print("Validation Loss: %f" % l)
    saver.save(sess, "model/model.ckpt")
