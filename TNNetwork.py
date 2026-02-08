# import keras
import numpy as np
import Generator
import NoiseDetectFunctions as Ndf
import SupportFunctions as Sf
import ParallelGenerator
import tensorflow as tf
from keras import layers
from keras import activations
from matplotlib import pyplot as plt
import OnehotFunctions as Oh
import os
import torch
import pickle


def generate_nn(size):
    # this function is to generate the neural network with certain structure
    # size is the number of nodes in each layer
    # layer_number is the total number of layers (including the output layer
    # the default activation function is tanh
    layer_number = len(size)
    model = tf.keras.models.Sequential()  # get an empty model
    for i in range(layer_number-1):
        model.add(layers.Dense(size[i], activation='tanh'))
        # adding dense layers with tanh activation function
    model.add(layers.Dense(size[layer_number-1]))
    # adding output layer
    model.add(layers.LeakyReLU(alpha=1.0))
    # model.add(layers.Dense(size[layer_number-1], activation='softmax'))
    # the output layer do not need an activation function
    return model


def train_tn(npart, size):
    train_number = 10000  # number of train set states, mode = 0  # 153600
    dim = 2 ** npart  # dimension of Hilbert space in total, and we consider qubit system right now
    small_batch = 2 ** 6  # small batch size in training
    epoch = 1000  # total epoch in training
    max_dim = 4
    measure_number = 512
    length_of_label = 4 * (2 * max_dim + (npart - 2) * max_dim ** 2)

    path = '5-qubit512measure10000samples.pkl'
    with open(path, 'rb') as file:
        training_set = pickle.load(file)

    frequencies_train = torch.zeros([10000, measure_number])
    labels_train = torch.zeros([10000, length_of_label])
    fidelity_train = torch.zeros([10000])
    for i in range(10000):
        frequencies_train[i, :] = training_set[i][0:measure_number]
        labels_train[i, :] = training_set[i][measure_number:measure_number + length_of_label]
        fidelity_train[i] = training_set[i][-1]

    frequencies_train = frequencies_train.numpy()
    labels_train = labels_train.numpy()
    frequencies_train = tf.convert_to_tensor(frequencies_train, dtype=tf.float32)
    labels_train = tf.convert_to_tensor(labels_train, dtype=tf.float32)

    model = generate_nn(size)

    model.compile(optimizer='SGD',  #tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
                  loss='mse',
                  metrics=['accuracy'])

    path_part1 = "./checkpoint/"
    path_part2 = "_qubit_1000tn_oht"
    path_part3 = ".ckpt"
    npart_str = str(npart)

    checkpoint_save_path = path_part1 + npart_str + path_part2 + path_part3
    if os.path.exists(checkpoint_save_path + '.index'):
        print('------load the model------')
        model.load_weights(checkpoint_save_path)

    # cp_callback = keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_save_path, save_weights_only=True,
    #     monitor='accuracy', save_best_only=True
    # )
    history = model.fit(frequencies_train, labels_train, batch_size=small_batch, epochs=epoch,
                        validation_split=0.35, validation_freq=20)

