#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
#tanmoy
#import pdb
#import matplotlib.pyplot as plt
#import pylab

kbT = (8.617343E-5) * 300 
beta = 1.0 / kbT
N_grid = 100
f_cvt = 1./96.485
cv_dim = 2

class Reader(object):
    def __init__(self, config):
        # copy from config
        self.data_path = config.data_path
        #tanmoy
        self.chk_path = config.chk_path
        self.num_epoch = config.num_epoch
        self.batch_size = config.batch_size   

    def prepare(self):
        self.index_count = 0
        self.current_setindex = 0
        tr_data = np.loadtxt(self.data_path+'data.raw')
        tr_data[:,cv_dim:] *= f_cvt
        self.inputs_train = tr_data[:,:]
        self.inputs_test = tr_data[:5,:]
        # print(np.shape(self.inputs_train))
        self.current_train_size = self.inputs_train.shape[0]
        self.train_size = self.inputs_train.shape[0]
        self.n_input = cv_dim
    
    def sample_train(self):
        self.index_count += self.batch_size
        if self.index_count > self.current_train_size:
            # shuffle the data
            self.index_count = self.batch_size
            ind = np.random.choice(self.current_train_size, self.current_train_size, replace=False)
            self.inputs_train = self.inputs_train[ind, :]
        ind = np.arange(self.index_count-self.batch_size, self.index_count)
        return self.inputs_train[ind, :]    

    def all_train(self):
    	return self.inputs_train

class Model(object):
    def __init__(self, config, sess):
        self.sess = sess
        # copy from config
        self.data_path = config.data_path
        #tanmoy
        self.chk_path = config.chk_path
        self.n_neuron = config.n_neuron
        self.useBN = config.useBN
        self.n_displayepoch = config.n_displayepoch
        self.starter_learning_rate = config.starter_learning_rate
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate
        self.display_in_training = config.display_in_training
        self.restart = config.restart
        
    def train(self, reader):
        reader.prepare()
        self.n_input = reader.n_input
        #tanmoy
        self.chk_path = reader.chk_path
        self.inputs_train = tf.placeholder(tf.float64, [None, self.n_input + cv_dim], name='inputs')
        self.is_training = tf.placeholder(tf.bool)
        self._extra_train_ops = []
        self.global_step = tf.get_variable('global_step', [],
                                            initializer=tf.constant_initializer(1),
                                            trainable=False, dtype=tf.int32)
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,\
            self.decay_steps*reader.train_size//reader.batch_size,self.decay_rate, staircase=True)
        self.mv_decay = 1.0 - self.learning_rate/self.starter_learning_rate;

        self.energy, self.l2_loss, self.rel_error_k\
            = self.build_force (self.inputs_train, suffix = "test", reuse = False)

        # train operations
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.l2_loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=self.global_step, name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        saver = tf.train.Saver()
        #tanmoy
        chkfile = self.chk_path + 'model.ckpt'

        # initialization
        if (self.restart == False):
            sample_used = 0
            epoch_used = 0
            self.sess.run(tf.global_variables_initializer())
            print('# start training from scratch')
        else:
            self.sess.run(tf.global_variables_initializer())
            #tanmoy
            saver.restore(self.sess, chkfile)
            print("Model restored.")
            cur_step = self.sess.run(self.global_step)
            sample_used = cur_step * reader.batch_size
            epoch_used = sample_used // reader.train_size

        start_time = time.time()
        
        while epoch_used < reader.num_epoch:
            # print('# doing training')
            inputs_train = reader.sample_train()
            self.sess.run([self.train_op], 
                          feed_dict={self.inputs_train: inputs_train,
                                     self.is_training: True})
            sample_used += reader.batch_size
            #print(sample_used)
            if (sample_used // reader.train_size) > epoch_used:
                epoch_used = sample_used // reader.train_size
                if epoch_used % self.n_displayepoch == 0:
                    #tanmoy
                    save_path = saver.save(self.sess, chkfile)
                    error = self.sess.run(self.l2_loss,
                                            feed_dict={self.inputs_train: inputs_train,
                                            self.is_training: False})
                    error2 = np.mean(np.sqrt(self.sess.run(self.rel_error_k,
                                            feed_dict={self.inputs_train: inputs_train,
                                            self.is_training: False})))

                    current_lr = self.sess.run(tf.to_double(self.learning_rate))
                    if self.display_in_training:
                        print("epoch: %3u, ab_err: %.4e, rel_err: %.4e, lr: %.4e" % (epoch_used, np.sqrt(error), error2, current_lr))
                #if epoch_used % (10*self.n_displayepoch) == 0:
        np.savetxt('errors.dat', np.sqrt(self.sess.run(self.rel_error_k,
                                feed_dict={self.inputs_train: reader.inputs_train,
                                self.is_training: False})))
        
        # xx = pylab.linspace(-np.pi, np.pi, N_grid)
        # yy = pylab.linspace(-np.pi, np.pi, N_grid)
        # delta2 = 2.0 * np.pi / N_grid
        # delta2 = delta2 * delta2
        # zz = pylab.zeros([len(xx), len(yy)])
        # my_grid = np.zeros((N_grid*N_grid,2))
        # zero_grid = np.zeros((N_grid*N_grid,1))
        # zero_grid4 = np.zeros((N_grid*N_grid,4))
        # for i in range(N_grid):
        #     for j in range(N_grid):
        #         my_grid[i*N_grid + j, 0] = i
        #         my_grid[i*N_grid + j, 1] = j

        # for i in xrange(len(xx)):
        #     print(i)
        #     for j in xrange(len(yy)):
        #         #my_input = np.concatenate([zero_grid + xx[i], np.reshape(my_grid[:,0], [-1,1]), zero_grid + yy[j], np.reshape(my_grid[:,1], [-1,1]), zero_grid4], axis = 1)
        #         my_input = np.concatenate([zero_grid + xx[i], zero_grid + yy[j], my_grid, zero_grid4], axis = 1)
        #         zz[j, i] =  (kbT) * np.log(np.sum(delta2 * np.exp(-beta * self.sess.run(self.energy,
        #                                 feed_dict={self.inputs_train: np.reshape(my_input, [-1,8]),
        #                                 self.is_training:False}))))
        # np.savetxt('nn_FE.dat', np.matrix(zz) - np.min(zz))
        # pylab.pcolor(xx, yy, zz - np.min(zz))
        # pylab.colorbar()
        # pylab.savefig("myfig12.png")
        # pylab.show()
                    
        end_time = time.time()
        print("running time: %.3f s" % (end_time-start_time))

        # plotting
        #turned on by tanmoy
        #inputs_train = reader.all_train()
        #my_encoder = self.sess.run(self.encoder_out,
        #                            feed_dict={self.inputs_train: inputs_train,
        #                            self.is_training: False})
        #my_type = reader.time_tag;
        #plt.figure
        #plt.scatter(my_encoder[:, 0], my_encoder[:, 1], c=my_type)
        #plt.colorbar()
        #plt.savefig('time.png')
        #plt.show()
        
    def build_force (self, 
                     inputs, 
                     suffix, 
                     reuse = None):
        #tanmoy 1027
        dists = tf.slice(inputs, [0,0], [-1,cv_dim], name='dists')
        forces_hat = tf.slice(inputs, [0,cv_dim], [-1,cv_dim], name='forces')
        inputs = tf.concat([dists, tf.square(dists)], 1)
        layer = self._one_layer(inputs, self.n_neuron[0], name='layer_0', reuse=reuse)
        for ii in range(1,len(self.n_neuron)) :
            layer = self._one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii), reuse=reuse)
        energy_ = self._final_layer(layer, 1, activation_fn = None, name='energy', reuse=reuse)
        energy = tf.identity (energy_, name='o_energy')
        forces = - tf.reshape(tf.stack(tf.gradients(energy, dists)), [-1, cv_dim], name='o_forces')
        force_dif = forces_hat - forces
        forces_norm = tf.reshape(tf.reduce_sum(forces * forces, axis = 1), [-1, 1])
        forces_dif_norm = tf.reshape(tf.reduce_sum(force_dif * force_dif, axis = 1), [-1, 1])
        l2_loss = tf.reduce_mean(forces_dif_norm, name='l2_loss')
        rel_error_k = forces_dif_norm / (1E-8 + forces_norm)
        return energy, l2_loss, rel_error_k
        
    def _one_layer(self, 
                   inputs, 
                   outputs_size, 
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=None,
                   seed=None):
        with tf.variable_scope(name, reuse=reuse):
            shape = inputs.get_shape().as_list()
            w = tf.get_variable('matrix', 
                                [shape[1], outputs_size], 
                                tf.float64,
                                tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed))
            b = tf.get_variable('bias', 
                                [outputs_size], 
                                tf.float64,
                                tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed))
            hidden = tf.matmul(inputs, w) + b

        if activation_fn != None:
            if self.useBN:
                # None
                hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                return activation_fn(hidden_bn)
            else:
                return activation_fn(hidden)
        else:
            if self.useBN:
                # None
                return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden

    def _final_layer(self, 
                   inputs, 
                   outputs_size, 
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=None,
                   seed=None):
        with tf.variable_scope(name, reuse=reuse):
            shape = inputs.get_shape().as_list()
            w = tf.get_variable('matrix', 
                                [shape[1], outputs_size], 
                                tf.float64,
                                tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed))
            hidden = tf.matmul(inputs, w)

        if activation_fn != None:
            if self.useBN:
                # None
                hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                return activation_fn(hidden_bn)
            else:
                return activation_fn(hidden)
        else:
            if self.useBN:
                # None
                return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden
    

    def _batch_norm(self, x, name, reuse):
        """Batch normalization"""
        with tf.variable_scope(name, reuse=reuse):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, tf.float64,
                                   initializer=tf.random_normal_initializer(0.0, stddev=0.1, dtype=tf.float64))
            gamma = tf.get_variable('gamma', params_shape, tf.float64,
                                    initializer=tf.random_uniform_initializer(0.1, 0.5, dtype=tf.float64)) 
        with tf.variable_scope(name+'moving', reuse=False):
            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float64,
                                          initializer=tf.constant_initializer(0.0, tf.float64),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, tf.float64,
                                              initializer=tf.constant_initializer(1.0, tf.float64),
                                              trainable=False)
        # These ops will only be preformed when training
        mean, variance = tf.nn.moments(x, [0], name='moments')
        self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, self.mv_decay))
        self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, self.mv_decay))
        mean, variance = control_flow_ops.cond(self.is_training, 
                                               lambda: (mean, variance),
                                               lambda: (moving_mean, moving_variance))
#       # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
        y.set_shape(x.get_shape())
        return y
