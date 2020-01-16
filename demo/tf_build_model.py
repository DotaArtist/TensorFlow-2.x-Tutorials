#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tf 1.0 resnet build model"""

__author__ = 'yp'


import json
import os
import tensorflow as tf
import datetime
import numpy as np
import cv2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_STDDEV = 0.01


class BatchPreprocessor(object):

    def __init__(self, dataset_file_path, num_classes, output_size=[128, 128], horizontal_flip=False, shuffle=False,
                 mean_color=[132.2766, 139.6506, 146.9702], multi_scale=None, is_load_img=True):
        self.num_classes = num_classes
        self.output_size = output_size
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle
        self.mean_color = mean_color
        self.multi_scale = multi_scale

        self.pointer = 0
        self.images = []
        self.labels = []

        if is_load_img:
            # Read the dataset file
            dataset_file = open(dataset_file_path, encoding='utf-8')
            lines = dataset_file.readlines()

            # NoneType = type(None)
            counter = 0

            for line in lines:
                items = line.split(' ')
                # tmp = cv2.imread(items[0])
                self.images.append(items[0])
                self.labels.append(int(items[1]))
                counter += 1
                # if os.path.isfile(items[0]) and os.path.getsize(items[0]) > 1024:
                #     self.images.append(items[0])
                #     self.labels.append(int(items[1]))
                #     counter += 1
                # else:
                #     pass

                if counter % 10000 == 0:
                    pass
                    # print(counter)

            # Shuffle the data
            if self.shuffle:
                self.shuffle_data()

    def shuffle_data(self):
        images = self.images[:]
        labels = self.labels[:]
        self.images = []
        self.labels = []

        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:(self.pointer+batch_size)]
        labels = self.labels[self.pointer:(self.pointer+batch_size)]

        # Update pointer
        self.pointer += batch_size

        # Read images
        images = np.ndarray([batch_size, self.output_size[0], self.output_size[1], 3])

        for i in range(len(paths)):
            # img = cv2.imread(paths[i])
            img = cv2.imdecode(np.fromfile(paths[i], dtype=np.uint8), -1)
            # img = paths[i]

            # Flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            if self.multi_scale is None:
                # Resize the image for output
                img = cv2.resize(img, (self.output_size[0], self.output_size[0]))
                img = img.astype(np.float32)

            elif isinstance(self.multi_scale, list):
                # Resize to random scale
                new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]
                img = cv2.resize(img, (new_size, new_size))
                img = img.astype(np.float32)

                # random crop at output size
                diff_size = new_size - self.output_size[0]
                random_offset_x = np.random.randint(0, diff_size, 1)[0]
                random_offset_y = np.random.randint(0, diff_size, 1)[0]
                img = img[random_offset_x:(random_offset_x+self.output_size[0]),
                          random_offset_y:(random_offset_y+self.output_size[0])]

            # Subtract mean color
            img = np.array(img, dtype=np.float64)
            img -= np.array(self.mean_color)
            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.num_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        # Return array of images and labels
        return images, one_hot_labels, labels

    @staticmethod
    def process_single_img(img_path):
        try:
            images = np.ndarray([1, 224, 224, 3])
            img = cv2.imread(img_path).astype(np.float32)
            img = cv2.resize(img, (224, 224))
            img -= np.array([132.2766, 139.6506, 146.9702])
            images[0] = img
            return images
        except AttributeError:
            return None

    @staticmethod
    def process_batch_img(img_path, label_list, counter=0):
        images = np.ndarray([64, 224, 224, 3])
        name_list = []

        _index = 0
        for i in range(counter, len(img_path)):
            counter += 1
            _images = np.ndarray([64, 224, 224, 3])
            _label_list = []
            _name_list = []
            # img = cv2.imread(img_path[i]).astype(np.float32)
            img = cv2.imdecode(np.fromfile(img_path[i], dtype=np.uint8), -1)
            img = cv2.resize(img, (224, 224))
            img = np.array(img, dtype=np.float64)
            img -= np.array([132.2766, 139.6506, 146.9702])
            images[_index] = img
            _index += 1
            name_list.append(img_path[i])

            if len(name_list) >= 64:
                _images = images
                _name_list = name_list

                images = np.ndarray([64, 224, 224, 3])
                name_list = []
                _index = 0

                yield _images, _name_list

        if len(name_list) != 0:
            yield images, name_list


class ResNetModel(object):

    def __init__(self, is_training, depth=50, num_classes=1000):
        self.is_training = is_training
        self.num_classes = num_classes
        self.depth = depth

        if depth in NUM_BLOCKS:
            self.num_blocks = NUM_BLOCKS[depth]
        else:
            raise ValueError('Depth is not supported; it must be 50, 101 or 152')

    def inference(self, x):
        # Scale 1
        with tf.variable_scope('scale1', reuse=tf.AUTO_REUSE):
            s1_conv = conv(x, ksize=7, stride=2, filters_out=64)
            s1_bn = bn(s1_conv, is_training=self.is_training)
            s1 = tf.nn.relu(s1_bn)

        # Scale 2
        with tf.variable_scope('scale2', reuse=tf.AUTO_REUSE):
            s2_mp = tf.nn.max_pool(s1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            s2 = stack(s2_mp, is_training=self.is_training, num_blocks=self.num_blocks[0], stack_stride=1,
                       block_filters_internal=64)

        # Scale 3
        with tf.variable_scope('scale3', reuse=tf.AUTO_REUSE):
            s3 = stack(s2, is_training=self.is_training, num_blocks=self.num_blocks[1], stack_stride=2,
                       block_filters_internal=128)

        # Scale 4
        with tf.variable_scope('scale4', reuse=tf.AUTO_REUSE):
            s4 = stack(s3, is_training=self.is_training, num_blocks=self.num_blocks[2], stack_stride=2,
                       block_filters_internal=256)

        # Scale 5
        with tf.variable_scope('scale5', reuse=tf.AUTO_REUSE):
            s5 = stack(s4, is_training=self.is_training, num_blocks=self.num_blocks[3], stack_stride=2,
                       block_filters_internal=512)

        # post-net
        # avg_pool = tf.reduce_mean(s5, reduction_indices=[1, 2], name='avg_pool')

        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            self.avg_pool = tf.reduce_mean(s5, reduction_indices=[1, 2], name='avg_pool')
            self.prob, _weights, _bias = fc(self.avg_pool, num_units_out=self.num_classes, return_params=True)

        # return self.prob
        return self.prob, self.avg_pool, _weights, _bias

    def loss(self, batch_x, batch_y=None):
        y_predict, _avg, _w, _b = self.inference(batch_x)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([cross_entropy_mean] + regularization_losses)
        return self.loss

    def optimize(self, learning_rate, train_layers=[]):
        trainable_var_names = ['weights', 'biases', 'beta', 'gamma']
        var_list = [v for v in tf.trainable_variables() if
            v.name.split(':')[0].split('/')[-1] in trainable_var_names and
            contains(v.name, train_layers)]
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=var_list)

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([self.loss]))

        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        return tf.group(train_op, batchnorm_updates_op)

    def load_original_weights(self, session, skip_layers=[]):
        weights_path = 'D:/model_file/ResNet-L{}.npy'.format(self.depth)
        weights_dict = np.load(weights_path, encoding='bytes').item()

        for op_name in weights_dict:
            parts = op_name.split('/')

            # if contains(op_name, skip_layers):
            #     continue

            if parts[0] == 'fc' and self.num_classes != 1000:
                continue

            full_name = "{}:0".format(op_name)
            var = [v for v in tf.global_variables() if v.name == full_name][0]
            session.run(var.assign(weights_dict[op_name]))


"""
Helper methods
"""


def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)


def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


def stack(x, is_training, num_blocks, stack_stride, block_filters_internal):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, is_training, block_filters_internal=block_filters_internal, block_stride=block_stride)
    return x


def block(x, is_training, block_filters_internal, block_stride):
    filters_in = x.get_shape()[-1]

    m = 4
    filters_out = m * block_filters_internal
    shortcut = x

    with tf.variable_scope('a'):
        a_conv = conv(x, ksize=1, stride=block_stride, filters_out=block_filters_internal)
        a_bn = bn(a_conv, is_training)
        a = tf.nn.relu(a_bn)

    with tf.variable_scope('b'):
        b_conv = conv(a, ksize=3, stride=1, filters_out=block_filters_internal)
        b_bn = bn(b_conv, is_training)
        b = tf.nn.relu(b_bn)

    with tf.variable_scope('c'):
        c_conv = conv(b, ksize=1, stride=1, filters_out=filters_out)
        c = bn(c_conv, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut_conv = conv(x, ksize=1, stride=block_stride, filters_out=filters_out)
            shortcut = bn(shortcut_conv, is_training)

    return tf.nn.relu(c + shortcut)


def fc(x, num_units_out, return_params=False):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())

    if not return_params:
        return tf.nn.xw_plus_b(x, weights, biases)
    else:
        return tf.nn.xw_plus_b(x, weights, biases), weights, biases


def contains(target_str, search_arr):
    rv = False

    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break

    return rv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/mytrain.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/myval.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

tf.app.flags.DEFINE_integer('is_builder_model', 1, 'build model:1')
tf.app.flags.DEFINE_integer('is_get_feature', 0, 'get feature:1')

FLAGS = tf.app.flags.FLAGS

MODEL_VERSION = '1'
MODEL_PATH = '../export_base/'


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    is_training = tf.placeholder('bool', [])
    # is_training = tf.constant(True, dtype=tf.bool)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = ResNetModel(is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)

    _p, avg_pool, _f, _ = model.inference(x)
    saver = tf.train.Saver()

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    batch_preprocess = BatchPreprocessor(dataset_file_path="",
                                         num_classes=FLAGS.num_classes,
                                         output_size=[224, 224],
                                         horizontal_flip=True,
                                         shuffle=True, multi_scale=multi_scale
                                         , is_load_img=False)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, "../training/resnet_20191017_113334/checkpoint/model_epoch11.ckpt")

        if FLAGS.is_get_feature == 1:
            counter = 0

            image_list = os.listdir("/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/img")
            image_list = ["/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/img/" + i for i in image_list]
            inputs_batch = batch_preprocess.process_batch_img(image_list)

            for batch in inputs_batch:
                inputs, file_list = batch

                if (not isinstance(inputs, type(None))) and inputs.shape == (64, 224, 224, 3):
                    s, prob = sess.run([avg_pool, _p], feed_dict={x: inputs, is_training: False})

                    for file_index in range(len(file_list)):
                        _feature = s[file_index]
                        _feature = [round(_i, 6) for _i in _feature]
                        _prob = prob[file_index].tolist()
                        _pred = _prob.index(max(_prob))

                        with open("../feature_5_data_cate3/{0}.txt".format(str(_pred)), mode="a") as f1:
                            f1.writelines("{0}\t{1}\n".format(str(file_list[file_index][:-4]),
                                                              str(s[file_index].tolist()),
                                                              # str(prob[file_index].tolist())
                                                              ))
                            counter += 1

                    if counter % 1000 == 0:
                        print(counter)

        if FLAGS.is_builder_model == 1:
            # Export model
            # export_path_base = FLAGS.export_path_base
            # export_path = os.path.join(
            #     tf.compat.as_bytes(export_path_base),
            #     tf.compat.as_bytes(str(FLAGS.model_version)))
            # print('Exporting trained model to', export_path)

            # Build the signature_def_map.
            # output == tf.nn.xw_plus_b(x, weights, biases)
            builder = tf.saved_model.builder.SavedModelBuilder("".join([MODEL_PATH, MODEL_VERSION]))
            image_input = tf.saved_model.utils.build_tensor_info(x)
            # image_pooling = tf.saved_model.utils.build_tensor_info(avg_pool)
            is_train = tf.saved_model.utils.build_tensor_info(is_training)
            # fc_weight = tf.saved_model.utils.build_tensor_info(_w)
            # fc_bias = tf.saved_model.utils.build_tensor_info(_b)
            y_pred = tf.saved_model.utils.build_tensor_info(_p)
            # feature_center = tf.saved_model.utils.build_tensor_info(_f)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                  inputs={'image_input': image_input, 'is_train': is_train},
                  outputs={
                      # 'image_pooling': image_pooling,
                      # 'fc_weight': fc_weight,
                      # 'fc_bias': fc_bias,
                      'y_pred': y_pred
                      },
                  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
            )

            builder.save()
            print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
