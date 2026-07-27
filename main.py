import numpy as np
import Generator
import NoiseDetectFunctions as Ndf
import SupportFunctions as Sf
import ParallelGenerator
import tensorflow as tf
from keras import layers
from keras.layers import LeakyReLU
from keras.layers import Softmax
from keras import activations
from matplotlib import pyplot as plt
import OnehotFunctions as Oh
import os
import scipy.io


# only some functions are used
def performance(output_array, dimension, number, rhos):
    # this function is to evaluate the performance of the network
    # output is tensor format, from tensorflow's network
    # dimension is the dimension of the density matrix, d*d
    # number is the total number of the states,
    # rhos is the origin states that put into test
    # output_array = np.array(output)  # turn the tensor into ndarray
    re_rhos = np.zeros([dimension, dimension, number], dtype=complex)  # allocate space for reconstructed rhos
    fidelity = np.zeros([number], dtype=np.float32)  # allocate space for fidelity
    noiseparas = output_array[:, dimension**2:-1]
    id = np.eye(dimension) / dimension
    for j in range(number):
        alps = output_array[j, 0:].T  # get the output label
        # alps = alps_pre[0:para_number] + 1j * alps_pre[para_number:2*para_number]  # build the alpha vector
        r = Ndf.rebuild_r_label(alps, dimension)  # build R (after Cholesky decompose)
        [re_rho, tr] = Ndf.r_to_rho(r)  # build rho = R'*R
        noiseparas[j, :] = alps[dimension**2:-1]/tr
        # p = alps[dimension**2]
        re_rhos[0:, 0:, j] = re_rho  # save rebuild rho
        rho = rhos[0:, 0:, j]  # get the original rho
        fidelity[j] = Sf.fidelity(rho, re_rho)  # calculate fidelity
    mean_fidelity = np.mean(fidelity)  # get mean fidelity
    return fidelity, mean_fidelity, re_rhos, noiseparas


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
    model.add(LeakyReLU(alpha=1.0))
    # model.add(layers.Dense(size[layer_number-1], activation='softmax'))
    # the output layer do not need an activation function
    return model


def noise_nn_keras(npart, size):
    train_number = 7680  # number of train set states, mode = 0
    test_number = 30  # number of test set states, mode = 1
    dim = 2 ** npart  # dimension of Hilbert space in total, and we consider qubit system right now
    measure_times = 40  # actual measuring times, using in mode = 1
    small_batch = 2 ** 5  # small batch size in training
    epoch = 2000  # total epoch in training

    operators = Ndf.operator_generator(npart, dim)  # get the measuring operators, use Pauli operators now
    [train_f, train_labelF, train_rhos] = ParallelGenerator.generator(npart, train_number, 0, measure_times)
    # get the data set of training part, mode = 0

    train_label = train_labelF[0:dim**2, :]
    # cast the data structure into tensor flow type
    train_f = tf.cast(train_f.T, tf.float32)
    train_label = tf.cast(train_label.T, tf.float32)

    model = generate_nn(size)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1.8, momentum=0.0),
                  loss='mse',
                  metrics=['accuracy'])
    path_part1 = "./checkpoint/"
    path_part2 = "_qubit.ckpt"
    npart_str = str(npart)
    checkpoint_save_path = path_part1+npart_str+path_part2

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path, save_weights_only=True,
        monitor='accuracy', save_best_only=True
    )
    history = model.fit(train_f, train_label, batch_size=small_batch, epochs=epoch,
                        validation_split=0.3, validation_freq=20, workers=12, callbacks=[cp_callback])
    model.summary()
    # '''
    acc = history.history['accuracy']
    loss = history.history['loss']
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='training accuracy')
    plt.title('training accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='training loss')
    plt.title('training loss')
    plt.legend()
    plt.show()
    # '''


def continue_to_train(npart, size, mode, measure_times=10):
    train_number = 2360  # number of train set states, mode = 0
    test_number = 10  # number of test set states, mode = 1, code will be done later
    # npart = 2  # number of parties in system
    dim = 2 ** npart  # dimension of Hilbert space in total, and we consider qubit system right now
    # measure_times = 100  # actual measuring times, using in mode = 1
    small_batch = 2 ** 6  # small batch size in training
    epoch = 1800  # total epoch in training

    operators = Ndf.operator_generator(npart, dim)  # get the measuring operators, use Pauli operators now
    # [train_f, train_labelF, train_rhos] = Generator.generator(npart, dim, train_number, operators, measure_times, 0)
    [train_f, train_labelF, train_rhos] = ParallelGenerator.generator(npart, train_number, mode, measure_times)
    # get the data set of training part, mode = 0

    train_label = train_labelF[0:dim**2, :]
    # cast the data structure into tensor flow type
    train_f = tf.cast(train_f.T, tf.float32)
    train_label = tf.cast(train_label.T, tf.float32)

    model = generate_nn(size)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=2.9, momentum=0.0),
                  loss='mse',
                  metrics=['accuracy'])

    path_part1 = "./checkpoint/"
    path_part2 = "_qubit.ckpt"
    npart_str = str(npart)
    checkpoint_save_path = path_part1 + npart_str + path_part2
    # checkpoint_save_path = "./checkpoint/three_qubit.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('------load the model------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path, save_weights_only=True,
        monitor='accuracy', save_best_only=True
    )
    history = model.fit(train_f, train_label, batch_size=small_batch, epochs=epoch,
                        validation_split=0.35, validation_freq=20, callbacks=[cp_callback], workers=12)


def train_oh(npart, size, mode, one_hot_sections=10, measure_times=10):
    train_number = 153600  # number of train set states, mode = 0
    dim = 2 ** npart  # dimension of Hilbert space in total, and we consider qubit system right now
    small_batch = 2 ** 6  # small batch size in training
    epoch = 2000  # total epoch in training

    [train_f, train_labelF, train_rhos] = ParallelGenerator.generator(npart, train_number, mode, measure_times)
    # get the data set of training part, mode = 0

    train_label = train_labelF[0:dim**2, :]
    train_label_oh = np.zeros([dim**2 * (one_hot_sections+1), train_number])
    table = Oh.table_generator(dim)
    for i in range(train_number):
        train_label_oh[:, i] = Oh.to_one_hot(train_label[:, i], dim, one_hot_sections, table)
    # cast the data structure into tensor flow type
    train_f = tf.cast(train_f.T, tf.float32)
    train_label = tf.cast(train_label_oh.T, tf.float32)

    model = generate_nn(size)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=3.2, momentum=0.0),  #tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
                  loss='mse',
                  metrics=['accuracy'])

    path_part1 = "./checkpoint/"
    path_part2 = "_qubit_2000c6_oh"
    path_part3 = ".ckpt"
    npart_str = str(npart)
    measure_times_str = str(mode * measure_times)
    one_hot_sections_str = str(one_hot_sections)
    checkpoint_save_path = path_part1 + npart_str + path_part2 + one_hot_sections_str + "N" + measure_times_str + path_part3
    if os.path.exists(checkpoint_save_path + '.index'):
        print('------load the model------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path, save_weights_only=True,
        monitor='accuracy', save_best_only=True
    )
    history = model.fit(train_f, train_label, batch_size=small_batch, epochs=epoch,
                        validation_split=0.35, validation_freq=20, callbacks=[cp_callback], workers=12)


def predict_function(npart, size, mode, measure_times):
    train_number = 8000  # number of train set states, mode = 0
    dim = 2 ** npart  # dimension of Hilbert space in total, and we consider qubit system right now

    operators = Ndf.operator_generator(npart, dim)  # get the measuring operators, use Pauli operators now
    # [train_f, train_labelF, train_rhos] = Generator.generator(npart, dim, train_number, operators, measure_times, 0)
    [train_f, train_labelF, train_rhos] = ParallelGenerator.generator(npart, train_number, mode, measure_times)
    # get the data set of training part, mode = 0

    train_label = train_labelF[0:dim**2, :]
    # cast the data structure into tensor flow type
    train_f = tf.cast(train_f.T, tf.float32)
    train_label = tf.cast(train_label.T, tf.float32)

    model = generate_nn(size)

    path_part1 = "./checkpoint/"
    path_part2 = "_qubit.ckpt"
    npart_str = str(npart)
    model_save_path = path_part1 + npart_str + path_part2
    # model_save_path = "./checkpoint/three_qubit.ckpt"
    model.load_weights(model_save_path)

    alpsnn = model.predict(train_f)
    alpsnn_array = np.array(alpsnn)  # turn the tensor into ndarray
    [dist, mean_dist, rerhos, noiseparas] = performance(alpsnn_array, dim, train_number, train_rhos)

    # x = range(train_number)
    f1 = plt.figure(1)
    plt.title(npart_str+'-qubit')
    # plt.xlabel('white noise, or depolarising noise')
    # plt.xlabel('average phase flip noise on qubits')
    plt.xlabel('fidelity with noisy state')
    plt.ylabel('fidelity with reconstructed state')
    # x = range(train_number)
    # plt.ylabel('reconstructed noise')
    # x = alpsnn_array[:, dim**2]
    x = train_labelF.T[:, dim**2]
    '''
    temp = np.zeros([train_number])
    for i in range(1+npart*3):
        temp += train_labelF.T[:, dim ** 2 + i]
    temp = temp/(1+npart*3)
    x = temp
    # '''
    # x = np.max(train_labelF.T[:, dim**2+1:dim**2+1+npart*3], axis=1)
    '''
    temp = np.zeros([train_number])
    for i in range(npart):
        temp += train_labelF.T[:, dim ** 2 + 1 + npart * 2 + i]
    temp = temp/npart
    x = temp
    # '''
    # x = (train_label[:, dim ** 2 + 1] + train_label[:, dim ** 2 + 2] + train_label[:, dim ** 2 + 3])/3
    y = dist
    # y = noiseparas[:, 1]
    z = np.polyfit(x, y, 20)
    p = np.poly1d(z)
    plt.plot(x, p(x), '.', color='gold')
    plt.scatter(x, y, s=2, color='cornflowerblue')
    # # plt.plot(alpsnn_array[:, -1], dist, '.', color='royalblue')

    # dist2 = np.zeros([train_number, 2], dtype=np.float32)
    # dist2[:, 0] = dist
    # dist2[:, 1] = dist

    # plt.boxplot(dist, vert=True)

    plt.show()
    # mean_dist = np.mean(dist)
    print(mean_dist)
    model.summary()


def predict_function_oh(npart, size, mode, one_hot_sections=10, measure_times=10):
    train_number = 100  # number of train set states, mode = 0
    dim = 2 ** npart  # dimension of Hilbert space in total, and we consider qubit system right now

    operators = Ndf.operator_generator(npart, dim)  # get the measuring operators, use Pauli operators now
    # [train_f, train_labelF, train_rhos] = Generator.generator(npart, dim, train_number, operators, measure_times, 0)
    [train_f, train_labelF, train_rhos] = ParallelGenerator.generator(npart, train_number, mode, measure_times)
    # get the data set of training part, mode = 0

    train_label = train_labelF[0:dim**2, :]
    # cast the data structure into tensor flow type
    train_f = tf.cast(train_f.T, tf.float32)
    train_label = tf.cast(train_label.T, tf.float32)

    model = generate_nn(size)

    '''
    path_part1 = "./checkpoint/"
    path_part2 = "_qubit_2000c4_oh"
    path_part3 = ".ckpt"
    npart_str = str(npart)
    measure_times_str = str(mode * measure_times)
    one_hot_sections_str = str(one_hot_sections)
    model_save_path = path_part1 + npart_str + path_part2 + one_hot_sections_str + "N" + measure_times_str + path_part3
    model.load_weights(model_save_path)

    alpsnn = model.predict(train_f)
    alpsnn_array_oh = np.array(alpsnn)  # turn the tensor into ndarray
    # '''
    # '''
    path_part1 = "./checkpoint/"
    path_part2 = "_qubit_2000c5_oh"
    path_part3 = ".ckpt"
    npart_str = str(npart)
    measure_times_str = str(mode * measure_times)
    one_hot_sections_str = str(one_hot_sections)
    model_save_path = path_part1 + npart_str + path_part2 + one_hot_sections_str + "N" + measure_times_str + path_part3
    model.load_weights(model_save_path)
    alpsnn1 = model.predict(train_f)
    path_part2 = "_qubit_2000c2_oh"
    model_save_path = path_part1 + npart_str + path_part2 + one_hot_sections_str + "N" + measure_times_str + path_part3
    model.load_weights(model_save_path)
    alpsnn2 = model.predict(train_f)
    path_part2 = "_qubit_2000c3_oh"
    model_save_path = path_part1 + npart_str + path_part2 + one_hot_sections_str + "N" + measure_times_str + path_part3
    model.load_weights(model_save_path)
    alpsnn3 = model.predict(train_f)
    path_part2 = "_qubit_2000c4_oh"
    model_save_path = path_part1 + npart_str + path_part2 + one_hot_sections_str + "N" + measure_times_str + path_part3
    model.load_weights(model_save_path)
    alpsnn4 = model.predict(train_f)
    path_part2 = "_qubit_2000c6_oh"
    model_save_path = path_part1 + npart_str + path_part2 + one_hot_sections_str + "N" + measure_times_str + path_part3
    model.load_weights(model_save_path)
    alpsnn5 = model.predict(train_f)
    alpsnn = (alpsnn1 + alpsnn2 + alpsnn3 + alpsnn4)/4
    alpsnn_array_oh = np.array(alpsnn)  # turn the tensor into ndarray
    alpsnn_array_oh1 = np.array(alpsnn1)
    alpsnn_array_oh2 = np.array(alpsnn5)
    # '''
    alpsnn_array = np.zeros([train_number, dim**2])
    alpsnn_array1 = np.zeros([train_number, dim ** 2])
    alpsnn_array2 = np.zeros([train_number, dim ** 2])
    table = Oh.table_generator(dim)
    for i in range(train_number):
        alpsnn_array[i, :] = Oh.to_alps(alpsnn_array_oh[i, :], dim, one_hot_sections, table)
        alpsnn_array1[i, :] = Oh.to_alps(alpsnn_array_oh1[i, :], dim, one_hot_sections, table)
        alpsnn_array2[i, :] = Oh.to_alps(alpsnn_array_oh2[i, :], dim, one_hot_sections, table)
    [dist, mean_dist, rerhos, noiseparas] = performance(alpsnn_array, dim, train_number, train_rhos)
    [dist1, mean_dist1, rerhos1, noiseparas1] = performance(alpsnn_array1, dim, train_number, train_rhos)
    [dist2, mean_dist2, rerhos2, noiseparas2] = performance(alpsnn_array2, dim, train_number, train_rhos)

    f1 = plt.figure(1)
    plt.title(npart_str+'-qubit')
    plt.xlabel('fidelity with noisy state')
    plt.ylabel('fidelity with reconstructed state')
    x = train_labelF.T[:, dim**2]
    # y = dist
    # z = np.polyfit(x, y, 20)
    # p = np.poly1d(z)
    # plt.plot(x, p(x), '.', color='gold')
    # plt.scatter(x, y, s=2, color='cornflowerblue')
    y1 = dist1
    y2 = dist2
    plt.plot(x, y1, '.', color='gold')
    plt.plot(x, y2, '.', color='cornflowerblue')

    plt.show()
    print(mean_dist)
    model.summary()


def predict_function_and_return_fidelity(npart, size, number_of_states, mode, measure_times=10):
    # number_of_states = 5000  # number of train set states
    dim = 2 ** npart  # dimension of Hilbert space in total, and we consider qubit system right now
    operators = Ndf.operator_generator(npart, dim)  # get the measuring operators, use Pauli operators now
    [train_f, train_labelF, train_rhos] = ParallelGenerator.generator(npart, number_of_states, mode, measure_times)

    train_label = train_labelF[0:dim ** 2, :]
    # train_label = train_labelF
    # cast the data structure into tensor flow type
    train_f = tf.cast(train_f.T, tf.float32)
    train_label = tf.cast(train_label.T, tf.float32)

    model = generate_nn(size)

    path_part1 = "./checkpoint/"
    path_part2 = "_qubit.ckpt"
    npart_str = str(npart)
    model_save_path = path_part1 + npart_str + path_part2
    # model_save_path = "./checkpoint/three_qubit.ckpt"
    model.load_weights(model_save_path)

    alpsnn = model.predict(train_f)
    alpsnn_array = np.array(alpsnn)  # turn the tensor into ndarray
    [fidelity, mean_fidelity, rerhos, noise_paras] = performance(alpsnn_array, dim, number_of_states, train_rhos)
    return fidelity


if __name__ == '__main__':
    nparties = 2  # number of parties in system
    dim = 2 ** nparties  # dimension of Hilbert space in total, and we consider qubit system right now
    one_hot_sections = 10
    # size_of_nn = [4200, 4100, 4000, dim ** 2]
    # size_of_nn = [200, 200, 200, 200, dim ** 2]
    # size_of_nn = [380, 360, 340, dim**2]
    # size_of_nn = [280, 270, 260, dim**2+1+nparties*1]
    # size_of_nn = [100, 100, 100, dim ** 2]
    # size_of_nn = [120, 100, 80, dim**2+1+nparties*3]
    # size_of_nn = [120, 100, 80, dim ** 2]
    # size_of_nn = [20, 18, 16, dim**2+1+nparties*3]
    # layer_number_of_nn = 6
    # size_of_nn = [40, 30, 30, 1 + nparties * 3]

    # number_of_states = 2500
    # dists = np.zeros([number_of_states, 4], dtype=np.float32)
    # noise_nn_keras(nparties, size_of_nn)
    # continue_to_train(nparties, size_of_nn, 0, 32)
    # predict_function(nparties, size_of_nn, 0, 10)

    # continue_to_train(nparties, size_of_nn, 0, 100)
    # dists[:, 0] = predict_function_and_return_fidelity(nparties, size_of_nn, number_of_states, 0, 100)
    # continue_to_train(nparties, size_of_nn, 1, 32)
    # dists[:, 1] = predict_function_and_return_fidelity(nparties, size_of_nn, number_of_states, 1, 316)
    # continue_to_train(nparties, size_of_nn, 1, 100)
    # dists[:, 2] = predict_function_and_return_fidelity(nparties, size_of_nn, number_of_states, 1, 316)
    # continue_to_train(nparties, size_of_nn, 1, 316)
    # dists[:, 3] = predict_function_and_return_fidelity(nparties, size_of_nn, number_of_states, 1, 316)

    # f1 = plt.figure(1)
    # plt.title('2-qubit')
    # plt.xlabel('white noise, or depolarising noise')
    # plt.xlabel('average phase flip noise on qubits')
    # plt.xlabel('measuring times')
    # plt.ylabel('fidelity')
    # plt.boxplot(dists, vert=True)

    size_of_nn_oh = [100, 100, 100, dim**2*(one_hot_sections+1)]  # 2-qubit one hot
    # size_of_nn_oh = [400, 380, 360, dim ** 2 * (one_hot_sections + 1)]  # 3-qubit one hot
    # size_of_nn_oh = [1200, 1100, 1000, dim ** 2 * (one_hot_sections + 1)]  # 4-qubit one hot
    # size_of_nn_oh = [4000, 3600, 3200, dim ** 2 * (one_hot_sections + 1)]  # 5-qubit one hot
    # train_oh(nparties, size_of_nn_oh, 0, one_hot_sections)
    predict_function_oh(nparties, size_of_nn_oh, 0, one_hot_sections)

    plt.show()