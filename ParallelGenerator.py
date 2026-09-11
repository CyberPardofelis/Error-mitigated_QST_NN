import time
from multiprocessing import Pool
import numpy as np
import NoiseDetectFunctions as Ndf
import SupportFunctions as Sf
from numpy import random
from scipy import linalg
from scipy.linalg import sqrtm


def generate_state(npart, mode):
    if mode == 'GHZ':
        psi = Sf.ghz_state(npart)
    if mode=='W':
        psi = Sf.w_state(npart)
    if mode=='random_pure':
        psi = Sf.random_pure_state(npart)
    if mode=='GHZ_like':
        theta = random.random(1) * np.pi/2
        psi = Sf.ghz_like_state(npart, theta)
    if mode=='Dicke':
        a = int(random.random(1)*(npart-1))+1
        # a = 1
        psi = Sf.dicke_state(npart, a)

    return psi


class State:
    def __init__(self, npart, wn, bf, pf, ad, mix_flag=True):
        self.npart = npart  # number of parties
        self.dimension = 2 ** self.npart  # number of dimension

        # self.density_matrix = generate_state(npart, 'GHZ')
        self.wn = wn  # get scale of white noise, a number
        self.bf = bf  # get scale of bit flip noise, a npart*1 array
        self.pf = pf  # get scale of phase flip noise, a npart*1 array
        self.ad = ad  # get scale of amplitude damping noise, a npart*1 array
        self.mix_flag = True

        mode = 'Dicke'

        # percentage = 0.15
        percentage = 0
        a = random.random(1)
        if mix_flag:
            if a >= percentage:
                self.density_matrix = Sf.random_density_matrix(self.dimension)  # random density matrix
                self.alps = self.generate_alps()
            else:
                self.psi = generate_state(npart, mode)
                self.density_matrix = self.psi * self.psi.T.conj()
                self.alps = self.generate_alps_pure(self.psi)
                self.mix_flag = False
        else:
            self.psi = generate_state(npart, mode)
            self.density_matrix = self.psi * self.psi.T.conj()
            self.alps = self.generate_alps_pure(self.psi)


        self.noisy_matrix = self.density_matrix[:, :]  # initialize of noisy density matrix
        self.noisy_matrix = self.add_noise()

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

        '''
        # add some noise without any structure
        noise_matrix = Sf.random_density_matrix(self.dimension)
        weight_factor = np.sqrt(random.uniform(0, 1))
        noise_strength = 0.4 * weight_factor
        n_rho = self.density_matrix * (1-noise_strength) + noise_matrix * noise_strength
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

    def generate_alps_pure(self,psi):
        alps = np.zeros([self.dimension**2], dtype=np.float32)
        count = 0
        r = np.zeros([self.dimension, self.dimension], dtype=np.complex64)
        r[0, :] = psi.T.conj()
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

    def get_label(self, wnf=True, bff=False, pff=False, adf=False, noisy=True):
        length = self.dimension ** 2 + 1
        '''
        if wnf:
            length = length + 1
        if bff:
            length = length + self.npart
        if pff:
            length = length + self.npart
        if adf:
            length = length + self.npart
        if noisy:
            length = length + self.dimension ** 2
        '''
        label = np.zeros([length], dtype=np.float32)
        label[0:length - 1] = self.alps
        if self.mix_flag:
            label[length-1] = self.fidelity(self.noisy_matrix)
        else:
            label[length-1] = self.fidelity_pure(self.noisy_matrix)
        '''
        index = self.dimension ** 2
        if wnf:
            label[index] = self.fidelity(self.noisy_matrix)  # In fact, it's calculating the fidelity
            index = index + 1
        if bff:
            label[index:index+self.npart] = self.bf
            index = index + self.npart
        if pff:
            label[index:index + self.npart] = self.pf
            index = index + self.npart
        if adf:
            label[index:index+self.npart] = self.ad
        if noisy:
            label[index:index+self.dimension**2] = self.generate_noisy_alps()
        # '''
        return label

    def fidelity(self, rho):
        f = np.trace(sqrtm(np.dot(np.dot(sqrtm(rho), self.density_matrix), sqrtm(rho))))
        # f=tr(sqrt(sqrt(r1)*r2*sqrt(r2)))
        return f.real

    def fidelity_pure(self, rho):
        f=np.sqrt(np.dot(self.psi.T.conj(), np.dot(rho, self.psi)))
        return f.real


class MeasureParameters:
    def __init__(self, rho, operators, p_operators, mode, measure_times=10):
        self.rho = rho
        self.operators = operators
        self.mode = mode
        self.measure_times = measure_times
        self.p_operators = p_operators


def measure(measure_parameters):
    npart = measure_parameters.rho.npart
    frequencies = np.zeros([4 ** npart - 1], dtype=np.float32)
    # 2026-09-10 fix: was np.zeros([4 ** npart * 2 + npart * 3]); the new get_label() returns
    # only dim^2 + 1 elements, so we only need 4**npart + dim^2 = 4**npart*2 slots.
    # The old '+ npart*3' was for the legacy 23-element label (incl. noise params) that is no
    # longer produced by State.get_label(). Kept as a comment for historical reference.
    output = np.zeros([4 ** npart * 2])
    rhoo = measure_parameters.rho.density_matrix
    measure_parameters.rho.add_noise()
    rhon = measure_parameters.rho.noisy_matrix
    for i in range(0, 4 ** npart - 1):
        op = measure_parameters.operators[0:, 0:, i]
        p = np.real(np.trace(np.dot(op, rhon)))
        if measure_parameters.mode == 0:
            frequencies[i] = p
        if measure_parameters.mode == 1:
            count = 0
            rns = random.random(size=(measure_parameters.measure_times, 1))
            for j in range(measure_parameters.measure_times):
                if rns[j] <= p:
                    count += 1
            frequencies[i] = count/measure_parameters.measure_times
    output[0:4 ** npart - 1] = frequencies
    # 2026-09-10 fix: was output[4 ** npart - 1:4 ** npart * 3] (slice 15:48 for 2q),
    # which was BOTH out of bounds (output is only 38 long) AND mismatched with the
    # new get_label() length (17 elements vs 23 slots) — raised
    #   ValueError: could not broadcast input array from shape (17,) into shape (23,)
    # Old (working) TF_bkup shape was output[4**npart-1:4**npart*2 + npart*3] for the
    # 23-element get_label(). New label is dim^2+1, so the correct slice is
    # output[4**npart-1 : 4**npart*2], matching generator_sim's slice (line ~511).
    # output[4 ** npart - 1:4 ** npart * 2 + npart * 3] = measure_parameters.rho.get_label()  # legacy
    output[4 ** npart - 1:4 ** npart * 2] = measure_parameters.rho.get_label()
    return output


def measure_sim(measure_parameters):
    npart = measure_parameters.rho.npart
    # length = measure_parameters.operators.shape[2]
    length = 4 ** npart - 1
    frequencies = np.zeros([length], dtype=np.float32)
    # length_of_labels = 4 ** npart + 1 + 3 * npart
    length_of_labels = 4 ** npart + 1
    output = np.zeros([length_of_labels + length])
    measure_parameters.rho.add_noise()
    rhon = measure_parameters.rho.noisy_matrix
    if measure_parameters.mode == 0:
        for i in range(length):
            op = measure_parameters.operators[0:, 0:, i]
            frequencies[i] = np.real(np.trace(np.dot(op, rhon)))
    if measure_parameters.mode == 1:
        for i in range(length):
            op = measure_parameters.operators[0:, 0:, i]
            p = np.real(np.trace(np.dot(op, rhon)))
            count = 0
            rns = random.random(size=(measure_parameters.measure_times, 1))
            for j in range(measure_parameters.measure_times):
                if rns[j] <= p:
                    count += 1
            frequencies[i] = count / measure_parameters.measure_times
    if measure_parameters.mode == 2:
        p_rhos = np.zeros([length], dtype=np.float32)
        p_operators = measure_parameters.p_operators
        counts = np.zeros([length], dtype=int)
        numbers = np.zeros([length], dtype=int)
        for i in range(length):
            op = measure_parameters.operators[0:, 0:, i]
            p_rhos[i] = np.real(np.trace(np.dot(op, rhon)))
        random_numbers = random.random(size=(measure_parameters.measure_times * length, 2))
        for i in range(measure_parameters.measure_times * length):
            temp = random_numbers[i, 0]
            j = 0
            temp -= p_operators[j]
            while temp > 0:
                temp -= p_operators[j]
                j += 1
            numbers[j] += 1
            if p_rhos[j] >= random_numbers[i, 1]:
                counts[j] += 1
        frequencies = counts / numbers
    output[0:length] = frequencies
    # output[length:length+length_of_labels] = measure_parameters.rho.get_label()
    output[length:length + length_of_labels] = measure_parameters.rho.get_label()
    # output[length] = Sf.fidelity(measure_parameters.rho.density_matrix, rhon)
    return output


def measure_sim_test(rho, operators, p_operators, measure_times):
    length = operators.shape[2]
    p_rhos = np.zeros([length], dtype=np.float32)
    counts = np.zeros([length], dtype=int)
    numbers = np.zeros([length], dtype=int)
    frequencies = np.zeros([length], dtype=np.float32)
    for i in range(0, length):
        p_rhos[i] = np.real(np.trace(np.dot(operators[:, :, i], rho)))
    random_numbers = random.random(size=(measure_times, 2))
    for i in range(measure_times):
        temp = random_numbers[i, 0]
        j = 0
        temp -= p_operators[j]
        while temp > 0:
            temp -= p_operators[j]
            j += 1
        numbers[j] += 1
        if p_rhos[j] >= random_numbers[i, 1]:
            counts[j] += 1
    frequencies = counts/numbers
    return frequencies


def generator(npart, number, mode, measure_times=10):
    # time1 = time.time()
    # -----------------generate basic data---------------------
    dim = 2 ** npart
    rhos = np.zeros([dim, dim, number], dtype=complex)
    # ad = int(dim * (dim + 1) / 2)
    # 2026-09-10 fix: was np.zeros([dim ** 2 + 1 + npart * 3, number]) (e.g. 23 for 2q);
    # reduced to dim^2 + 1 (e.g. 17 for 2q) to match the new get_label() return length.
    label = np.zeros([dim ** 2 + 1, number], dtype=float)
    frequencies = np.zeros([4 ** npart - 1, number], dtype=np.float32)
    # allocate space

    '''
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

    # '''
    lb = 0.0
    ub = 0.0
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

    operators = Ndf.operator_generator(npart, dim)
    p_operators = np.ones([4 ** npart - 1], dtype=np.float32)
    p_operators = p_operators/(4 ** npart - 1)

    # -------------------set parallel parameters and so on----------------------
    para_pool = Pool(6)
    measure_parameters = []
    for i in range(number):
        state = State(npart, wns[i], bfs[:, i], pfs[:, i], ads[:, i])
        state.generate_alps()
        measure_parameter = MeasureParameters(state, operators, p_operators, mode, measure_times)
        measure_parameters.append(measure_parameter)
    fre_and_label = para_pool.map(measure, measure_parameters)
    # time2 = time.time()
    # print("time cost {}".format(time2-time1))
    # print(type(fre_and_label))
    r = np.zeros([dim, dim], dtype=complex)
    for i in range(number):
        frequencies[:, i] = fre_and_label[i][0:4**npart-1]
        # 2026-09-10 fix: was fre_and_label[i][4**npart-1:4**npart*2+3*npart] (23-slot slice);
        # trimmed to 4**npart*2 to match the new 17-element label from get_label().
        label[:, i] = fre_and_label[i][4**npart-1:4**npart*2]
        alps = label[0:4**npart, i]
        r = Ndf.rebuild_r_label(alps, dim)
        [rhos[:, :, i], tr] = Ndf.r_to_rho(r)
    return frequencies, label, rhos


def generator_sim(npart, number, mode, measure_times=10, mix_flag=True):
    # time1 = time.time()
    # -----------------generate basic data---------------------
    dim = 2 ** npart
    rhos = np.zeros([dim, dim, number], dtype=complex)
    rhons = np.zeros([dim, dim, number], dtype=complex)
    # ad = int(dim * (dim + 1) / 2)
    label = np.zeros([dim ** 2 + 1, number], dtype=float)
    frequencies = np.zeros([4 ** npart - 1, number], dtype=np.float32)
    # allocate space

    '''
    ub = 0.2
    wns = random.random(size=number)
    wns = wns ** npart
    lb_wn = 0.
    ub_wn = ub
    wns = wns * (ub_wn - lb_wn) + lb_wn
    # wns is the amount of white noise
    # lb_wn = lower bound of white noise, ub_wn = upper bound of white noise

    bfs = random.random(size=(npart, number))
    bfs = bfs ** npart
    lb_bf = 0.
    ub_bf = ub
    bfs = bfs * (ub_bf - lb_bf) + lb_bf
    # bfs is the amount of bit flip noise,
    # lb_bf = lower bound of bit flip, ub_bf = upper bound of bit flip

    pfs = random.random(size=(npart, number))
    pfs = pfs ** npart
    lb_pf = 0.
    ub_pf = ub
    pfs = pfs * (ub_pf - lb_pf) + lb_pf
    # bfs is the amount of phase flip noise,
    # lb_bf = lower bound of phase flip, ub_bf = upper bound of phase flip

    ads = random.random(size=(npart, number))
    ads = ads ** npart
    lb_ad = 0.
    ub_ad = ub
    ads = ads * (ub_ad - lb_ad) + lb_ad
    # bfs is the amount of amplitude damping noise,
    # lb_bf = lower bound of amplitude damping, ub_bf = upper bound of amplitude damping
    # '''

    # '''
    lb = 0.0
    ub = 0.08
    noise_parameters = random.random(size=(1 + 3 * npart, number))
    mean_parameters = np.mean(noise_parameters, 0)
    normalizer = random.random(size=number) * (ub - lb) + lb
    normalized_parameters = np.zeros([1 + npart * 3, number], dtype=np.float32)
    for i in range(number):
        normalized_parameters[:, i] = noise_parameters[:, i] * normalizer[i] / mean_parameters[i]

    wns = normalized_parameters[0, :]
    bfs = normalized_parameters[1:1 + npart, :]
    pfs = normalized_parameters[1 + npart:1 + 2 * npart, :]
    ads = normalized_parameters[1 + 2 * npart:1 + 3 * npart, :]
    # '''

    [operators, p_operators] = Ndf.operator_generator_with_p(npart, dim)

    # -------------------set parallel parameters and so on----------------------
    para_pool = Pool(6)
    measure_parameters = []
    for i in range(number):
        state = State(npart, wns[i], bfs[:, i], pfs[:, i], ads[:, i], mix_flag)
        # state.generate_alps()
        measure_parameter = MeasureParameters(state, operators, p_operators, mode, measure_times)
        measure_parameters.append(measure_parameter)
    fre_and_label = para_pool.map(measure_sim, measure_parameters)
    # time2 = time.time()
    # print("time cost {}".format(time2-time1))
    # print(type(fre_and_label))
    r = np.zeros([dim, dim], dtype=complex)
    rn = np.zeros([dim, dim], dtype=complex)
    for i in range(number):
        frequencies[:, i] = fre_and_label[i][0:4**npart-1]
        # label[:, i] = fre_and_label[i][4**npart-1:4**npart*2+3*npart]
        label[:, i] = fre_and_label[i][4 ** npart - 1:4 ** npart * 2]
        alps = label[0:4**npart, i]
        r = Ndf.rebuild_r_label(alps, dim)
        [rhos[:, :, i], tr] = Ndf.r_to_rho(r)
        # alpsn = label[4**npart+1:4**npart*2+1, i]
        # rn = Ndf.rebuild_r_label(alpsn, dim)
        # [rhons[:, :, i], tr] = Ndf.r_to_rho(rn)
    return frequencies, label, rhos, rhons
