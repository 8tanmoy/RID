#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from model import Reader, Model

class Config(object):
    batch_size = 16
    n_displayepoch = 500
    # use Batch Normalization or not
    useBN = False#True
    num_epoch = 12500
    n_neuron = [48, 24, 12]
    starter_learning_rate = 0.001
    decay_steps = 50
    decay_rate = 0.96
    data_path = './data/'
    #tanmoy
    chk_path = './'
    restart = False    
    display_in_training = True
    
def main():
    tf.reset_default_graph()
    with tf.Session() as sess:
        config = Config()
        print("Begin to optimize")
        reader = Reader(config)
        model = Model(config, sess)
        
        model.train(reader)



if __name__ == '__main__':
    main()
