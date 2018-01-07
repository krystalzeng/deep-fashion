from net_model import *
from pretrained_net_model import *
from image_dataset import BatchGenerator, get_fashion_dataset
from tensorflow.contrib.slim.nets import vgg
import tensorflow.contrib.slim as slim
import os
import pickle

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_iter', 10000, 'Total training iter')
flags.DEFINE_integer('step', 100, 'Save after ... iteration')
flags.DEFINE_bool('is_vgg', False, 'Whether to use the pretrained vgg-16 or the custom cnn')
flags.DEFINE_float('margin', 0.9, 'Contrastive loss margin')
flags.DEFINE_integer('embedding_size', 100, 'Embedding output size')
flags.DEFINE_bool('is_test_run', False, 'Whether we just test functionality')

FLAGS._parse_flags()

print('Batch size is ', FLAGS.batch_size)
print('Train iterations is ', FLAGS.train_iter)
print('Margin is ', FLAGS.margin)
print('Embedding size is ', FLAGS.embedding_size)
print('Is vgg is ', FLAGS.is_vgg)
print('Is test run ', FLAGS.is_test_run)

###############################################
# 1. Create the data generator
###############################################
dataset, image_dict = get_fashion_dataset('list_dataset.pickle', '../full/', test_run=FLAGS.is_test_run)
gen = BatchGenerator(dataset, image_dict, testset_proportion=0.1, batch_size=FLAGS.batch_size)

##################################################################
# 2. Define the training procedure
##################################################################
left = tf.placeholder(tf.float32, [None, 150, 150, 3], name='left')
right = tf.placeholder(tf.float32, [None, 150, 150, 3], name='right')

# required for vgg
# resized_left = tf.image.resize_images(left, [224, 224])
# resized_right = tf.image.resize_images(right, [224, 224])

with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
    label = tf.to_float(label)

margin = FLAGS.margin

left_output = mynet(left, reuse=False, embedding_size=FLAGS.embedding_size)
right_output = mynet(right, reuse=True, embedding_size=FLAGS.embedding_size)

# with slim.arg_scope(vgg.vgg_arg_scope()):
#     left_output_vgg = vgg_16(resized_left, num_classes=100, is_training=True, reuse=False)
#     right_output_vgg = vgg_16(resized_right, num_classes=100, is_training=True, reuse=True)

global_step = tf.Variable(0, trainable=False)

# if FLAGS.is_vgg:
#     loss = contrastive_loss(left_output_vgg, right_output_vgg, label, margin)
#     init_fn = slim.assign_from_checkpoint_fn(
#         os.path.join('vgg-16/', 'vgg_16.ckpt'),
#         slim.get_model_variables('vgg_16'))
# else:
loss = contrastive_loss(left_output, right_output, label, margin)

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

saver = tf.train.Saver()

######################################################################
# 3. Run the model
######################################################################
best_validation_loss = 100000.0
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
train_losses = []
dev_losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train iter
    for i in range(FLAGS.train_iter):
        total_loss = 0.0
        for _ in range(gen.total_batches):
            b_l, b_r, b_label = gen.next_batch()

            _, tl = sess.run([train_step, loss],
                                         feed_dict={left: b_l, right: b_r, label: b_label})
            total_loss += tl
        train_losses.append(total_loss / gen.total_batches)
        print("\r#%d - Loss" % i, total_loss / gen.total_batches)

        t_l, t_r, t_label = gen.test_set()
        # generate test
        total_dev_loss = 0.0
        for j in range(max(int(len(t_l) / gen.batch_size), 1)):
            lower = j * gen.batch_size
            upper = min((i + 1) * gen.batch_size, len(t_l))
            l = sess.run(loss, feed_dict={left: t_l[lower:upper], right: t_r[lower:upper], label: t_label[lower:upper]})
            total_dev_loss += l

        total_dev_loss = total_dev_loss / (max(int(len(t_l) / gen.batch_size), 1))
        dev_losses.append(total_dev_loss)
        print("Validation Loss: %f" % (total_dev_loss))
        if total_dev_loss < best_validation_loss:
            best_validation_loss = total_dev_loss
            saver.save(sess, "vgg={}-{}-{}-model/model.ckpt".format(FLAGS.is_vgg, FLAGS.embedding_size, FLAGS.margin))
        pickle.dump(train_losses, open('vgg={}-{}-{}-train_losses.pickle'.format(FLAGS.is_vgg, FLAGS.embedding_size, FLAGS.margin), 'wb'))
        pickle.dump(dev_losses, open('vgg={}-{}-{}-dev_losses.pickle'.format(FLAGS.is_vgg, FLAGS.embedding_size, FLAGS.margin), 'wb'))
