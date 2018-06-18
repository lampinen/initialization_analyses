from __future__ import print_function
from __future__ import division 

#from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plot

###### configuration ###########################################################

config = {
    "num_runs": 30,
    "batch_size": 10,
    "base_learning_rate": 0.001,
    "base_lr_decay": 0.8,
    "base_lr_decays_every": 10,
    "base_lr_min": 0.00001,
    "base_training_epochs": 200,
    "output_path": "./results/",
    "nobias": False, # no biases
    "linear": False,
    "num_val": 10000,
    "noise_prob": 0.2, # probability of flipping image pixels 
    "verbose": True,
    "layer_sizes": [256, 128, 64, 128, 256]
}

inits = [1.0, 0.1] # multiplies xavier initializer 

###### MNIST data loading and manipulation #####################################
# downloaded from https://pjreddie.com/projects/mnist-in-csv/

train_data = np.loadtxt("../SWIL/MNIST_data/mnist_train.csv", delimiter = ",")
test_data = np.loadtxt("../SWIL/MNIST_data/mnist_test.csv", delimiter = ",")

def process_data(dataset):
    """Get data split into dict with labels and images"""
    labels = dataset[:, 0]
    images = dataset[:, 1:]/255.
    flip_mask = np.random.binomial(1, config["noise_prob"], np.shape(images))
    images = (1.-flip_mask) * images + flip_mask * (1.-images)
    images = np.clip(images, 0., 1.);
    data = {"labels": labels, "images": images}
    return data

train_data = process_data(train_data)
test_data = process_data(test_data)

val_data = {"labels": train_data["labels"][:config["num_val"]],
            "images": train_data["images"][:config["num_val"], :]}
train_data = {"labels": train_data["labels"][config["num_val"]:],
              "images": train_data["images"][config["num_val"]:, :]}

###### Build model func ########################################################

def softmax(x, T=1):
    """Compute the softmax function at temperature T"""
    if T != 1:
        x /= T
    x -= np.amax(x)
    x = np.exp(x)
    x /= np.sum(x)
    if not(np.any(x)): # handle underflow
        x = np.ones_like(x)/len(x) 
    return x

def to_unit_rows(x):
    """Converts row vectors of a matrix to unit vectors"""
    return x/np.expand_dims(np.sqrt(np.sum(x**2, axis=1)), -1)

def _display_image(x):
    x = np.reshape(x, [28, 28])
    plot.figure()
    plot.imshow(x, vmin=0, vmax=1)

class MNIST_autoenc(object):
    """MNIST autoencoder architecture"""

    def __init__(self, layer_sizes, init_multiplier=1.0):
        """Create a MNIST_autoenc model. 
           layer_sizes: list of the hidden layer sizes of the model
           init_multiplier: multiplicative factor on the Xavier initializer
        """
        self.base_lr = config["base_learning_rate"]

        self.input_ph = tf.placeholder(tf.float32, [None, 784])
        self.lr_ph = tf.placeholder(tf.float32)

        self.bottleneck_size = min(layer_sizes)

	# small weight initializer
	weight_init = tf.contrib.layers.variance_scaling_initializer(factor=init_multiplier, mode='FAN_AVG')
	if config["linear"]:
	    intermediate_activation_fn=None
	    final_activation_fn=None
	else:
	    intermediate_activation_fn=tf.nn.relu
	    final_activation_fn=tf.nn.sigmoid
	

        net = self.input_ph
	bottleneck_layer_i = len(layer_sizes)//2
        for i, h_size in enumerate(layer_sizes):
	    if config["nobias"]:
	      net = slim.layers.fully_connected(net, h_size, activation_fn=intermediate_activation_fn,
						weights_initializer=weight_init,
						biases_initializer=None)
	    else:
	      net = slim.layers.fully_connected(net, h_size, activation_fn=intermediate_activation_fn,
						weights_initializer=weight_init)
            if i == bottleneck_layer_i: 
                self.bottleneck_rep = net
	if config["nobias"]:
	    self.output = slim.layers.fully_connected(net, 784, activation_fn=final_activation_fn,
						      weights_initializer=weight_init,
						      biases_initializer=None)
	else:
	    self.output = slim.layers.fully_connected(net, 784, activation_fn=final_activation_fn,
						      weights_initializer=weight_init)
						  
        self.loss = tf.nn.l2_loss(self.output-self.input_ph)

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        self.train = self.optimizer.minimize(tf.reduce_mean(self.loss))

        self.first_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fully_connected/weights')[0]
        self.first_biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fully_connected/biases')[0]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run_training(self, dataset, nepochs=100, log_file_prefix=None, val_dataset=None, test_dataset=None):
        """Train the model on a dataset"""
        with open(config["output_path"] + log_file_prefix + "losses.csv", "w") as fout:
            fout.write("epoch, train, val, test\n")
            train_loss = self.eval(dataset)
            val_loss = self.eval(dataset)
            test_loss = self.eval(dataset)
            fout.write("0, %f, %f, %f\n" % (train_loss,
                                            val_loss,
                                            test_loss))

            batch_size = config["batch_size"]
            for epoch in range(1, nepochs + 1):
                order = np.random.permutation(len(dataset["labels"]))
                for batch_i in xrange(len(dataset["labels"])//batch_size):
                    this_batch_images = dataset["images"][order[batch_i*batch_size:(batch_i+1)*batch_size], :]
                    self.sess.run(self.train, feed_dict={
                            self.input_ph: this_batch_images,
                            self.lr_ph: self.base_lr 
                        })

		train_loss = self.eval(dataset)
		val_loss = self.eval(val_dataset)
		test_loss = self.eval(test_dataset)
                fout.write("%i, %f, %f, %f\n" % (epoch,
                                                 train_loss,
                                                 val_loss,
                                                 test_loss))
                if config["verbose"]:
                    print("%i, %f, %f, %f\n" % (epoch,
                                                train_loss,
                                                val_loss,
                                                test_loss))

                # update lr
                if epoch > 0 and epoch % config["base_lr_decays_every"] == 0 and self.base_lr > config["base_lr_min"]: 
                    self.base_lr *= config["base_lr_decay"]
         

    def get_reps(self, images):
        """Gets bottleneck reps for the given images"""
        batch_size = config["batch_size"]
        reps = np.zeros([len(images), self.bottleneck_size])
        for batch_i in xrange((len(images)//batch_size) + 1):
            this_batch_images = images[batch_i*batch_size:(batch_i+1)*batch_size, :]
            reps[batch_i*batch_size:(batch_i+1)*batch_size, :] = self.sess.run(
                self.bottleneck_rep, feed_dict={
                    self.input_ph: this_batch_images 
                })
        return reps

    def get_loss(self, images):
        """Gets losses for the given images"""
        batch_size = config["batch_size"]
        loss = np.zeros([len(images)])
        for batch_i in xrange((len(images)//batch_size) + 1):
            this_batch_images = images[batch_i*batch_size:(batch_i+1)*batch_size, :]
            loss[batch_i*batch_size:(batch_i+1)*batch_size] = self.sess.run(
                self.loss, feed_dict={
                    self.input_ph: this_batch_images
                })
        return loss

    def eval(self, dataset):
        """Evaluates model on the given dataset. Returns average loss."""
        losses = self.get_loss(dataset["images"])
        losses_summarized = np.sum(losses)/len(dataset["labels"])#[np.sum(losses[dataset["labels"] == i])/np.sum(dataset["labels"] == i) for i in range(10)]
        return losses_summarized

    def display_output(self, image):
        """Runs an image and shows comparison"""
        output_image = self.sess.run(self.output, feed_dict={
                self.input_ph: np.expand_dims(image, 0) 
            })

        _display_image(image)
        _display_image(output_image)
        plot.show()



###### Run stuff ###############################################################

for run in range(config["num_runs"]):
    for init in inits:
        filename_prefix = "init%.2f_run%i_" %(init, run)
        print(filename_prefix)
        np.random.seed(run)
        tf.set_random_seed(run)

        model = MNIST_autoenc(layer_sizes=config["layer_sizes"],
                              init_multiplier=init)

        order = np.random.permutation(len(train_data["labels"]))
        train_data["labels"] = train_data["labels"][order]
        train_data["images"] = train_data["images"][order]

        model.run_training(train_data,
                           config["base_training_epochs"],
                           log_file_prefix=filename_prefix,
                           val_dataset=val_data,
                           test_dataset=test_data)

        tf.reset_default_graph()
