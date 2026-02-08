import random
from numpy import random
import numpy as np
import SupportFunctions as Sf
import NoiseDetectFunctions as Ndf
from numba import jit


def generator(npart, dim, number, operators, measure_times, mode):
    # npart is the number of particles, dim is the dimension of total hilbert space
    # number is the number of rhos, operators in the measuring operators
    # mode is the running mode of data generator, 0=identical mode, 1=actual measurement
    # measure_times is the measuring times for each operator in actual measurement
    rhos = np.zeros([dim, dim, number], dtype=complex)
    # ad = int(dim * (dim + 1) / 2)
    alps = np.zeros([dim ** 2, number], dtype=float)
    frequencies = np.zeros([4**npart-1, number], dtype=np.float32)
    # allocate space

    wns = random.random(size=number)
    lb_wn = 0.
    ub_wn = 0.5
    wns = wns * (ub_wn - lb_wn) + lb_wn
    # wns is the amount of white noise
    # lb_wn = lower bound of white noise, ub_wn = upper bound of white noise

    bfs = random.random(size=(npart, number))
    lb_bf = 0.
    ub_bf = 0.4
    bfs = bfs * (ub_bf - lb_bf) + lb_bf
    # bfs is the amount of bit flip noise,
    # lb_bf = lower bound of bit flip, ub_bf = upper bound of bit flip

    pfs = random.random(size=(npart, number))
    lb_pf = 0.
    ub_pf = 0.
    pfs = pfs * (ub_pf - lb_pf) + lb_pf
    # bfs is the amount of phase flip noise,
    # lb_bf = lower bound of phase flip, ub_bf = upper bound of phase flip

    ads = random.random(size=(npart, number))
    lb_ad = 0.
    ub_ad = 0.
    ads = ads * (ub_ad - lb_ad) + lb_ad
    # bfs is the amount of amplitude damping noise,
    # lb_bf = lower bound of amplitude damping, ub_bf = upper bound of amplitude damping

    # main loop
    for i in range(0, number):
        rho = Sf.random_density_matrix(dim)  # generate random state
        rhos[0:, 0:, i] = rho
        r = Sf.chol(rho)  # Cholesky decompose
        alps[0:, i] = Ndf.get_alps_label(r, dim)  # reshape R and get a vector that ready to use as labels
        # from generate rho to get alps, it's not a very slow process, so most time is on next line
    for i in range(number):
        par1 = wns[i]
        par2 = bfs[:, i]
        par3 = pfs[:, i]
        par4 = ads[:, i]
        frequencies[0:, i] = Ndf.measure(npart, dim, operators, rhos[:, :, i],
                                         measure_times, mode, par1, par2, par3, par4)
        # get the frequencies from measure function
    '''
    label = np.zeros([2*ad+1, number], dtype=np.float32)  # allocate space
    label[0:ad:1, 0:] = alps.real
    label[ad:2*ad:1, 0:] = alps.imag
    label[2*ad, 0:] = nps
    # divide the real part and imaginary part, make it a set of float number
    '''
    label = np.zeros([dim ** 2 + 1 + npart*1, number], dtype=np.float32)  # allocate space
    label[0:dim ** 2:1, 0:] = alps
    label[dim ** 2, 0:] = wns
    label[dim ** 2 + 1:dim ** 2 + 1 + npart, :] = bfs
    # label[dim ** 2 + 1 + npart:dim ** 2 + 1 + npart * 2, :] = pfs
    # label[dim ** 2 + 1 + npart * 2:dim ** 2 + 1 + npart * 3, :] = ads
    return frequencies, label, rhos


def test_generator(number):
    x = random.random(size=number)
    noise = random.random(size=number) * 0.05 - 0.025
    y = np.sin(x)+noise
    return x, y
