import time
from multiprocessing import Pool
import numpy as np
import NoiseDetectFunctions as Ndf
import SupportFunctions as Sf
from numpy import random
from scipy import linalg
from scipy.linalg import sqrtm


class State:
    def __init__(self, npart, wn, bf, pf, ad):
        self.npart = npart  # number of parties
        self.dimension = 2 ** self.npart  # number of dimension
        self.density_matrix = Sf.random_density_matrix(self.dimension)  # random density matrix
        self.wn = wn  # get scale of white noise, a number
        self.bf = bf  # get scale of bit flip noise, a npart*1 array
        self.pf = pf  # get scale of phase flip noise, a npart*1 array
        self.ad = ad  # get scale of amplitude damping noise, a npart*1 array
        self.noisy_matrix = self.density_matrix[:, :]  # initialize of noisy density matrix
        self.noisy_matrix = self.add_noise()
        # self.alps = np.zeros([self.dimension ** 2], dtype=np.float32)
        self.alps = self.generate_alps()
        self.noisy_alps = self.generate_noisy_alps()
        self.noise_fidelity = self.fidelity(self.noisy_matrix)

    def add_noise(self, wnf=True, bff=True, pff=True, adf=True):
        # wnf, bff, pff, and adf are Boolean variable, default set as True
        # '''
        # add some noise with certain structure
        if wnf:
            whitenoise = np.eye(self.dimension) / self.dimension
            self.noisy_matrix = self.noisy_matrix * (1 - self.wn) + whitenoise * self.wn
        identity = np.eye(2)
        bf = Sf.pauli(1)
        pf = Sf.pauli(3)
        n_rho = np.zeros([self.dimension, self.dimension], dtype=complex)
        if bff and not pff:
            for ind in range(2 ** self.npart):
                tempo = np.array([[1]], dtype=complex)
                tempp = 1.
                for n in range(self.npart):
                    if ind & (2 ** n) != 0:
                        tempo = np.kron(tempo, bf)
                        tempp = tempp * self.bf[n]
                    else:
                        tempo = np.kron(tempo, identity)
                        tempp = tempp * (1 - self.bf[n])
                n_rho = n_rho + tempp * np.dot(np.dot(tempo, self.noisy_matrix), tempo)
            self.noisy_matrix = n_rho
        if not bff and pff:
            for ind in range(2 ** self.npart):
                tempo = np.array([[1]], dtype=complex)
                tempp = 1.
                for n in range(self.npart):
                    if ind & (2 ** n) != 0:
                        tempo = np.kron(tempo, pf)
                        tempp = tempp * self.pf[n]
                    else:
                        tempo = np.kron(tempo, identity)
                        tempp = tempp * (1 - self.pf[n])
                n_rho = n_rho + tempp * np.dot(np.dot(tempo, self.noisy_matrix), tempo)
            self.noisy_matrix = n_rho
        if bff and pff:
            for ind in range(4 ** self.npart):
                tempo = np.array([[1]], dtype=complex)
                tempp = 1.
                for n in range(self.npart):
                    tempo2 = np.eye(2)
                    if ind & (2 ** (n * 2)) != 0:
                        tempo2 = np.dot(tempo2, bf)
                        tempp = tempp * self.bf[n]
                    else:
                        tempo2 = np.dot(tempo2, identity)
                        tempp = tempp * (1 - self.bf[n])
                    if ind & (2 ** (n * 2) * 2) != 0:
                        tempo2 = np.dot(tempo2, pf)
                        tempp = tempp * self.pf[n]
                    else:
                        tempo2 = np.dot(tempo2, identity)
                        tempp = tempp * (1 - self.pf[n])
                    tempo = np.kron(tempo, tempo2)
                n_rho = n_rho + tempp * np.dot(np.dot(tempo, self.noisy_matrix), tempo)
            self.noisy_matrix = n_rho[:, :]
        if adf:
            e0 = np.zeros([2, 2, self.npart], dtype=complex)
            e1 = np.zeros([2, 2, self.npart], dtype=complex)
            for i in range(self.npart):
                e0[:, :, i] = np.array([[1., 0], [0., np.sqrt(1 - self.ad[i])]])
                e1[:, :, i] = np.array([[0., np.sqrt(self.ad[i])], [0., 0.]])
            n_rho = np.zeros([self.dimension, self.dimension], dtype=complex)
            for ind in range(2 ** self.npart):
                tempo = np.array([[1.]])
                for n in range(self.npart):
                    if ind & 2 ** n == 0:
                        tempo = np.kron(tempo, e0[:, :, n])
                    else:
                        tempo = np.kron(tempo, e1[:, :, n])
                n_rho = n_rho + np.dot(np.dot(tempo, self.noisy_matrix), tempo.T)
            self.noisy_matrix = n_rho[:, :]
        # '''

        return n_rho

    def generate_alps(self):
        alps = np.zeros([self.dimension**2], dtype=np.float32)
        count = 0
        r = linalg.cholesky(self.density_matrix)
        for i in range(self.dimension):
            alps[count:count+i+1:1] = r.real[0:i+1:1, i]
            count += i+1
        for i in range(1, self.dimension):
            alps[count:count+i:1] = r.imag[0:i:1, i]
            count += i
        self.alps = alps
        return alps

    def generate_noisy_alps(self):
        alps = np.zeros([self.dimension ** 2], dtype=np.float32)
        count = 0
        r = linalg.cholesky(self.noisy_matrix)
        for i in range(self.dimension):
            alps[count:count + i + 1:1] = r.real[0:i + 1:1, i]
            count += i + 1
        for i in range(1, self.dimension):
            alps[count:count + i:1] = r.imag[0:i:1, i]
            count += i
        self.alps = alps
        return alps

    def get_label(self):
        d = self.dimension ** 2
        label = np.zeros([d * 2 + 1], dtype=np.float32)
        label[0:d] = self.alps
        label[d:d * 2] = self.generate_noisy_alps()
        label[d * 2] = self.noise_fidelity
        return label

    def fidelity(self, rho):
        f = np.trace(sqrtm(np.dot(np.dot(sqrtm(rho), self.density_matrix), sqrtm(rho))))
        # f=tr(sqrt(sqrt(r1)*r2*sqrt(r2)))
        return f.real


class Operators:
    def __init__(self, npart, mode=0, measure_times=10):
        self.npart = npart
        dim = 2 ** npart
        self.operators = Ndf.operator_generator(npart, dim)
        self.mode = mode
        self.measure_times = measure_times


def measure_p(operators):
    npart = operators.npart
    lb = 0.01
    ub = 0.09

    wn = lb + (ub - lb) * random.random(1)
    bf = lb + (ub - lb) * random.random(npart)
    pf = lb + (ub - lb) * random.random(npart)
    ad = lb + (ub - lb) * random.random(npart)

    rho = State(npart, wn, bf, pf, ad)
    frequencies = np.zeros([4 ** npart - 1], dtype=np.float32)
    output = np.zeros([4 ** npart * 2 + npart * 3])
    rhon = rho.noisy_matrix
    for i in range(0, 4 ** npart - 1):
        op = operators.operators[0:, 0:, i]
        frequencies[i] = np.real(np.trace(np.dot(op, rhon)))
    output[0:4 ** npart - 1] = frequencies
    # output[4 ** npart - 1:4 ** npart * 2 + npart * 3] = measure_parameters.rho.get_label()
    output[4 ** npart - 1:4 ** npart * 3] = rho.get_label()
    return output


def generator(npart, number, mode, measure_times=10):
    # time1 = time.time()
    # -----------------generate basic data---------------------
    dim = 2 ** npart
    rhos = np.zeros([dim, dim, number], dtype=complex)
    # ad = int(dim * (dim + 1) / 2)
    label = np.zeros([dim ** 2 + 1 + npart * 3, number], dtype=float)
    frequencies = np.zeros([4 ** npart - 1, number], dtype=np.float32)
    # allocate space

    # '''
    lb = 0.01
    ub = 0.09
    wns = random.random(size=number)
    lb_wn = lb
    ub_wn = ub
    wns = wns * (ub_wn - lb_wn) + lb_wn
    # wns is the amount of white noise
    # lb_wn = lower bound of white noise, ub_wn = upper bound of white noise

    bfs = random.random(size=(npart, number))
    lb_bf = lb
    ub_bf = ub
    bfs = bfs * (ub_bf - lb_bf) + lb_bf
    # bfs is the amount of bit flip noise,
    # lb_bf = lower bound of bit flip, ub_bf = upper bound of bit flip

    pfs = random.random(size=(npart, number))
    lb_pf = lb
    ub_pf = ub
    pfs = pfs * (ub_pf - lb_pf) + lb_pf
    # bfs is the amount of phase flip noise,
    # lb_bf = lower bound of phase flip, ub_bf = upper bound of phase flip

    ads = random.random(size=(npart, number))
    lb_ad = lb
    ub_ad = ub
    ads = ads * (ub_ad - lb_ad) + lb_ad
    # bfs is the amount of amplitude damping noise,
    # lb_bf = lower bound of amplitude damping, ub_bf = upper bound of amplitude damping
    # '''

    '''
    lb = 0.0
    ub = 0.09
    noise_parameters = random.random(size=(1 + 3 * npart, number))
    mean_parameters = np.max(noise_parameters, 0)
    normalizer = random.random(size=number) * (ub - lb) + lb
    normalized_parameters = np.zeros([1 + npart * 3, number], dtype=np.float32)
    for i in range(number):
        normalized_parameters[:, i] = noise_parameters[:, i] * normalizer[i] / mean_parameters[i]

    wns = normalized_parameters[0, :]
    bfs = normalized_parameters[1:1+npart, :]
    pfs = normalized_parameters[1+npart:1 + 2*npart, :]
    ads = normalized_parameters[1+2*npart:1 + 3*npart, :]
    # '''

    operators = Operators(npart)

    # -------------------set parallel parameters and so on----------------------
    para_pool = Pool(8)

    fre_and_label = para_pool.map(measure_p, [operators] * number)
    # time2 = time.time()
    # print("time cost {}".format(time2-time1))
    # print(type(fre_and_label))
    r = np.zeros([dim, dim], dtype=complex)
    for i in range(number):
        frequencies[:, i] = fre_and_label[i][0:4**npart-1]
        label[:, i] = fre_and_label[i][4**npart-1:4**npart*2+3*npart]
        alps = label[0:4**npart, i]
        r = Ndf.rebuild_r_label(alps, dim)
        [rhos[:, :, i], tr] = Ndf.r_to_rho(r)
    return frequencies, label, rhos