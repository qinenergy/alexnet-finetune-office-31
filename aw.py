"""This is an TensorFLow implementation of AlexNet-Finetune Task on Office-31.
   Work done when writing thesis with Dr. Wen Li at ETH Zurich.

Blog: https://www.qin.ee/2018/06/25/da-office/

@author: Qin Wang (contact: wang (at) qin.ee )

"""




import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime

"""
Configuration Part.
"""
Iterator = tf.data.Iterator
valmode = True # Use full taget set for evaluation
display_step = 15

# Path to the textfiles for the trainings and validation set
source_file = 'amazon.txt'
target_file = 'webcam.txt'
size = len(open(source_file).readlines())

print("size", size)

# Learning params
learning_rate = 0.001
num_epochs = 400
print("EPOCH_NUM going to run", num_epochs) 
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 31
#train_layers = ['fc8', 'fc7', 'fc6', 'conv4', 'conv5', 'grl']


checkpoint_path = "./checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(source_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True, size=size)
    val_data = ImageDataGenerator(target_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False, fulval=valmode)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)

    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, ["fc8"])

# Link variable to model output
score = model.fc8
soft = model.soft

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss1 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
    
    l2_loss =  tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                    if 'bias' not in v.name])
    loss = loss1 + 0.0001/2 * l2_loss
# List of trainable variables of the layers we want to train
var_list1 = [v for v in tf.trainable_variables() if "grl" not in v.name and "bias" not in v.name]
var_list1b = [v for v in tf.trainable_variables() if "grl" not in v.name and "bias" in v.name]
var_list2 = [v for v in tf.trainable_variables() if "grl" in v.name and "bias" not in v.name]
var_list2b = [v for v in tf.trainable_variables() if "grl" in v.name and "bias" in v.name]

var_list = var_list1+var_list1b+var_list2+var_list2b
print(var_list)
# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer1 = tf.train.GradientDescentOptimizer(lr)
    optimizer1b = tf.train.GradientDescentOptimizer(lr*2)
    optimizer2 = tf.train.GradientDescentOptimizer(lr*10)
    optimizer2b = tf.train.GradientDescentOptimizer(lr*20)

    #train_op = optimizer1.apply_gradients(grads_and_vars=gradients)
    train_op1 = optimizer1.apply_gradients(grads_and_vars=gradients[:len(var_list1)])
    train_op1b = optimizer1b.apply_gradients(grads_and_vars=gradients[len(var_list1):len(var_list1+var_list1b)])
    train_op2 = optimizer2.apply_gradients(grads_and_vars=gradients[len(var_list1+var_list1b):len(var_list1+var_list1b+var_list2)])
    train_op2b = optimizer2b.apply_gradients(grads_and_vars=gradients[len(var_list1+var_list1b+var_list2):])
    train_op = tf.group(train_op1, train_op1b, train_op2, train_op2b)



# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accunum = tf.shape(y)[0]



# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.ceil(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        for step in range(train_batches_per_epoch):

            # get next batch of data
            p = epoch / num_epochs + step/train_batches_per_epoch/num_epochs
            x1, y1 = sess.run(next_batch)
            img_batch = x1 
            label_batch = y1 
            if epoch < num_epochs - 20:
                lr_ = 0.001 / (1. + 0.001 * p)**0.75
            else:
                lr_ = 0.0001
            _, loss1_ = sess.run([train_op, loss1], feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate,
                                          lr: lr_})
            if step % display_step == 0:
                print("train_loss1:", loss1_, lr_, p)
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        num = 0.
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc, num_ = sess.run([accuracy, accunum], feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            num += num_
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc/num))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
