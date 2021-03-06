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
    "base_learning_rates": [0.005, 0.0033],
    "base_learning_rate": None,
    "base_lr_decay": 0.8,
    "base_lr_decays_every": None,
    "decay_everys": [10, 3, 1],
    "base_lr_min": 0.0001,
    "base_training_epochs": 200,
    "output_path": "./results/",
    "nobias": False, # no biases
    "linear": False,
    "num_val": 10000,
    "noise_prob": 0.25, # probability of flipping image pixels 
    "noise_sd": 0.2, # sd of gaussian noise added
    "verbose": True,
    "layer_sizes": [1000, 1000, 1000, 1000],
    "stop_thresh": 0.001, # stop when trainin error reaches this
    "stop_val_increase_ratio": 1.05, # ratio of increase in validation error over min at which we stop
    "num_adv_examples": 10, # number of test examples to construct adversarial examples for
    "max_adv_iterations": 10000,
    "adv_eta": 0.05, # gradient descent step size for constructing adversarial examples
    "adv_momentum": 0.9 # momentum for adversarial example construction
}

inits = [1.0] # factor in xavier initializer 

###### MNIST data loading and manipulation #####################################
# downloaded from https://pjreddie.com/projects/mnist-in-csv/

np.random.seed(0)

train_data = np.loadtxt("../SWIL/MNIST_data/mnist_train.csv", delimiter = ",")
test_data = np.loadtxt("../SWIL/MNIST_data/mnist_test.csv", delimiter = ",")

def process_data(dataset):
    """Get data split into dict with labels and images"""
    labels = dataset[:, 0]
    images = dataset[:, 1:]/255.
    flip_mask = np.random.binomial(1, config["noise_prob"], np.shape(images))
    images = (1.-flip_mask) * images + flip_mask * (1.-images)
    images += config["noise_sd"] * np.random.standard_normal(np.shape(images))
    images = np.clip(images, 0, 1)
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

class MNIST_model(object):
    """MNIST autoencoder or classification architecture"""

    def __init__(self, layer_sizes, init_multiplier=1.0, model_type="autoencoder", regularization="None", reg_param=None):
        """Create a MNIST_model. 
           layer_sizes: list of the hidden layer sizes of the model
           init_multiplier: multiplicative factor on the Xavier initializer
        """
        self.classification = model_type == "classification"
        self.reg_type = regularization
        self.reg_param = reg_param

        self.base_lr = config["base_learning_rate"]

        self.input_ph = tf.placeholder(tf.float32, [None, 784])
        if model_type == "classification":
            self.target_ph = tf.placeholder(tf.int64, [None, ])

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
        with slim.arg_scope([slim.layers.fully_connected],
                            weights_regularizer=slim.l2_regularizer(reg_param) if regularization=="L2" else None):
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

            if model_type == "classification":
                if config["nobias"]:
                    self.logits = slim.layers.fully_connected(net, 10, activation_fn=None,
                                                              weights_initializer=weight_init,
                                                              biases_initializer=None)
                else:
                    self.logits = slim.layers.fully_connected(net, 10, activation_fn=None,
                                                              weights_initializer=weight_init)
                
                self.output = tf.nn.softmax(self.logits)
                                                          
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_ph,
                                                                           logits=self.logits)

                self.adv_class_ph = tf.placeholder(tf.int64, [None, ])
                adv_class_grads = tf.one_hot(self.adv_class_ph, depth=10)
                self.adv_grads = tf.gradients(xs=self.input_ph, ys=self.output, grad_ys=adv_class_grads)

            elif model_type == "autoencoder":
                if config["nobias"]:
                    self.output = slim.layers.fully_connected(net, 784, activation_fn=final_activation_fn,
                                                              weights_initializer=weight_init,
                                                              biases_initializer=None)
                else:
                    self.output = slim.layers.fully_connected(net, 784, activation_fn=final_activation_fn,
                                                              weights_initializer=weight_init)
                                                          
                self.loss = tf.nn.l2_loss(self.output-self.input_ph)


        all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '.*weights.*')
        if regularization == "Zeroing":
            self.squish_weights = [tf.assign(x, reg_param*x) for x in all_weights]
        elif regularization == "Truncating":
            def _truncate_SVD(X, tail_pct=0.1): 
                """Truncates SVD to only modes that are higher than specified percent of max mode strength"""
                S, U, V = tf.svd(X, full_matrices=False)
                tail_amount = tail_pct*tf.reduce_max(S)
                new_S = tf.where(tf.greater(S, tail_amount), S, tf.zeros_like(S)) 
                return tf.matmul(U, tf.matmul(tf.diag(new_S), tf.transpose(V))) 
            self.truncate_weights = [tf.assign(x, _truncate_SVD(x, tail_pct=reg_param)) for x in all_weights]
            #self.display_weight_S = [tf.svd(x, compute_uv=False) for x in all_weights]


        self.optimizer = tf.train.MomentumOptimizer(self.lr_ph, 0.9)
        self.train = self.optimizer.minimize(tf.reduce_mean(self.loss))

        self.first_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fully_connected/weights')[0]
        self.first_biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fully_connected/biases')[0]

        self.saver = tf.train.Saver()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
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
            min_val_loss = val_loss
            for epoch in range(1, nepochs + 1):
                order = np.random.permutation(len(dataset["labels"]))
                for batch_i in range(len(dataset["labels"])//batch_size):
                    this_batch_indices = order[batch_i*batch_size:(batch_i+1)*batch_size]
                    this_batch_images = dataset["images"][this_batch_indices, :]
                    this_batch_feed_dict = {
                        self.input_ph: this_batch_images,
                        self.lr_ph: self.base_lr 
                    }
                    if self.classification:
                        this_batch_feed_dict[self.target_ph] =dataset["labels"][this_batch_indices] 

                    self.sess.run(self.train, feed_dict=this_batch_feed_dict)

                # zeroing/truncating if applicable
                if self.reg_type == "Zeroing":
                    self.sess.run(self.squish_weights)
                elif self.reg_type == "Truncating":
                    self.sess.run(self.truncate_weights)

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

                if val_loss <= min_val_loss:
                    self.saver.save(self.sess, './checkpoints/model-checkpoint')
                    min_val_loss = val_loss

                # early stop?
                if train_loss < config["stop_thresh"]:
                    print("Early stopping!")
                    self.saver.restore(self.sess, './checkpoints/model-checkpoint')
                    break

                if val_loss > config["stop_val_increase_ratio"] * min_val_loss:
                    print("Early stopping -- validation error increasing.")
                    self.saver.restore(self.sess, './checkpoints/model-checkpoint')
                    break

                # update lr
                if epoch > 0 and epoch % config["base_lr_decays_every"] == 0 and self.base_lr > config["base_lr_min"]: 
                    self.base_lr *= config["base_lr_decay"]

         

    def get_reps(self, images):
        """Gets bottleneck reps for the given images"""
        batch_size = config["batch_size"]
        reps = np.zeros([len(images), self.bottleneck_size])
        for batch_i in range((len(images)//batch_size) + 1):
            this_batch_images = images[batch_i*batch_size:(batch_i+1)*batch_size, :]
            reps[batch_i*batch_size:(batch_i+1)*batch_size, :] = self.sess.run(
                self.bottleneck_rep, feed_dict={
                    self.input_ph: this_batch_images 
                })
        return reps

    def get_loss(self, dataset):
        """Gets losses for the given images"""
        batch_size = config["batch_size"]
        loss = np.zeros([len(dataset["labels"])])
        for batch_i in range((len(dataset["labels"])//batch_size)):
            this_batch_indices = range(batch_i*batch_size,(batch_i+1)*batch_size)
            this_batch_images = dataset["images"][this_batch_indices, :]
            this_batch_feed_dict = {
                    self.input_ph: this_batch_images
                }
            if self.classification:
                this_batch_feed_dict[self.target_ph] = dataset["labels"][this_batch_indices] 
            loss[batch_i*batch_size:(batch_i+1)*batch_size] = self.sess.run(
                self.loss, feed_dict=this_batch_feed_dict)
        return loss

    def eval(self, dataset):
        """Evaluates model on the given dataset. Returns average loss."""
        losses = self.get_loss(dataset)
        losses_summarized = np.sum(losses)/len(dataset["labels"])#[np.sum(losses[dataset["labels"] == i])/np.sum(dataset["labels"] == i) for i in range(10)]
        return losses_summarized

    def construct_adversarial_examples(self, images, labels, adversarial_classes, filename=None, create_file=False):
        """Constructs adversarial examples for given image and classes."""
        if not self.classification:
            raise NotImplementedError("Cannot construct adversarial examples for a non-classification model")
        curr_images = np.copy(images)
        updates = np.zeros_like(images)
        images_not_done = np.ones_like(labels, dtype=np.bool)
        adv_eta = config["adv_eta"]

        for _ in range(config["max_adv_iterations"]):
            this_feed_dict = {
                    self.input_ph: curr_images,
                    self.target_ph: labels,
                    self.adv_class_ph: adversarial_classes
                }
            curr_grads, curr_softmaxes = self.sess.run([self.adv_grads, self.output], feed_dict=this_feed_dict)
            curr_hardmaxes = np.argmax(curr_softmaxes, axis=-1)
            images_not_done = np.not_equal(curr_hardmaxes, adversarial_classes)
            updates = (1-config["adv_momentum"]) * adv_eta * curr_grads[0] + config["adv_momentum"] * updates 
            curr_images[images_not_done, :] += updates[images_not_done, :] 
            if not np.any(images_not_done):
                break

        l2_dists = np.linalg.norm(curr_images - images, axis=-1)
        if filename is not None:
            with open(filename, "w" if create_file else "a") as fout:
                if create_file:
                    fout.write("index, original_class, new_class, failed, l2_dist\n")
                for i in range(len(labels)):
                    fout.write("%i, %i, %i, %i, %f\n" % (i, labels[i], adversarial_classes[i], images_not_done[i], l2_dists[i]))
        return curr_images, l2_dists

    def display_output(self, image):
        """Runs an image and shows comparison if autoencoder"""
        res = self.sess.run(self.output, feed_dict={
                self.input_ph: np.expand_dims(image, 0) 
            })

        _display_image(image)
        if self.model_type == "autoencoder":
            _display_image(res)
        else:
            print(res)
        plot.show()



###### Run stuff ###############################################################


for run in range(config["num_runs"]):
    for regularization in ["Truncating", "Zeroing", "L2", "None"]:
        if regularization == "None":
            reg_params = [0.]
        elif regularization == "L2":
            reg_params = [0.01, 0.001, 0.0001]
        elif regularization == "Zeroing":
            reg_params = [0.33, 0.5, 0.1, 0.033]
        elif regularization == "Truncating":
            reg_params = [0.1, 0.2, 0.05, 0.33]
        
        for reg_param in reg_params:
            for model_type in ["classification"]: 
                for base_lr in config["base_learning_rates"]:
                    for decay_every in config["decay_everys"]:
                        config["base_learning_rate"] = base_lr # hacky
                        config["base_lr_decays_every"] = decay_every 
                        for init in inits:
                            filename_prefix = "decay0.8_every%i_type%s_baselr%f_init%.2f_reg%s_param%.3f_run%i_" %(decay_every, model_type, base_lr, init, regularization, reg_param, run)
                            print(filename_prefix)
                            np.random.seed(run)
                            tf.set_random_seed(run)

                            model = MNIST_model(layer_sizes=config["layer_sizes"],
                                                init_multiplier=init,
                                                regularization=regularization,
                                                reg_param=reg_param,
                                                model_type=model_type)

                            order = np.random.permutation(len(train_data["labels"]))
                            train_data["labels"] = train_data["labels"][order]
                            train_data["images"] = train_data["images"][order]

                            nadv = config["num_adv_examples"]

                            model.run_training(train_data,
                                               config["base_training_epochs"],
                                               log_file_prefix=filename_prefix,
                                               val_dataset=val_data,
                                               test_dataset=test_data)

                            for adv_off in range(1, 10):
                                model.construct_adversarial_examples(test_data["images"][:nadv, :],
                                                                     test_data["labels"][:nadv],
                                                                     np.mod(test_data["labels"][:nadv]+adv_off, 10),
                                                                     filename=config["output_path"] + filename_prefix + "adversarial.csv",
                                                                     create_file=adv_off == 1)

                            tf.reset_default_graph()
