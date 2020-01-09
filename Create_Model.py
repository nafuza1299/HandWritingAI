# Copyright 2016 Niek Temme.
# Adapted form the on the MNIST expert tutorial by Google. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
Documentation at
http://niektemme.com/ @@to do

This script is based on the Tensoflow MNIST expert tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
"""

#import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#TRAINING->optimizing->evaluation
#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))#3 dimentional value
b = tf.Variable(tf.zeros([10]))#3 dimentional value
y = tf.nn.softmax(tf.matmul(x, W) + b)#refering to prev batch computation

def weight_variable(shape):#for conv
  initial = tf.truncated_normal(shape, stddev=0.1)#generate random value from truncanted shape
  return tf.Variable(initial)#3 dimentional variable representing the weight_variable

def bias_variable(shape):#for relu
  initial = tf.constant(0.1, shape=shape)#3 dimentional biased_variable
  return tf.Variable(initial)#create three dimensional variable


def conv2d(x, W):# convolutional function
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):# maxpool function
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

 #creates an empty tensor with all elements set to zero with a shape
W_conv1 = weight_variable([5, 5, 1, 32])# conv layer with 32 neural number #weight layer 5*5 filter
b_conv1 = bias_variable([32])#neural number 32

x_image = tf.reshape(x, [-1,28,28,1])#reshape to meet mnist size
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#conv1+relu1
h_pool1 = max_pool_2x2(h_conv1)#maxpool layer


W_conv2 = weight_variable([5, 5, 32, 64])#conv layer with 64 neural number
b_conv2 = bias_variable([64])#biased with  64 neural number

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#conv2+relu2
h_pool2 = max_pool_2x2(h_conv2)#maxpool

W_fc1 = weight_variable([7 * 7 * 64, 1024])#fully connected weight one initializing var
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#flat with wfc1 to refine
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#fully connected

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])#fully connected weight 2#classification
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#softmax probabilistic classification

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#adam algorithm optimizer->optimizing
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# acccuracy built in by tensorflow->evaluation


"""
Train the model and save the model to disk as a model2.ckpt file
file is stored in the same directory as this python script is started

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
#with tf.Session() as sess:
    #sess.run(init_op)
for i in range(20000):#train times/iteration
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

save_path = saver.save(sess, "model2dum.ckpt")
print ("Model saved in file: ", save_path)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



