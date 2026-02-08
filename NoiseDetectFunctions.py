import random
import SupportFunctions as Sf
import numpy as np
from numpy import random


def operator_generator(npart, dim):
    # generate measuring operators, npart is the number of particles and dim is
    # the total Hilbert space dimension.
    operators = np.zeros((dim, dim, 4 ** npart-1), dtype=complex)
    # allocate space
    pauli = np.zeros((2, 2, 4), dtype=complex)
    pauli[0:, 0:, 3] = np.array([[1., 0.], [0., 1.]], dtype=complex)
    pauli[0:, 0:, 1] = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    pauli[0:, 0:, 2] = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
    pauli[0:, 0:, 0] = np.array([[1., 0.], [0., 0.]], dtype=complex)
    # generate basic pauli measurement operators
    # pauli[0:, 0:, 0] = np.array([[1., 0.], [0., 1.]], dtype=complex)
    # pauli[0:, 0:, 1] = np.array([[0., 1.], [1., 0.]], dtype=complex)
    # pauli[0:, 0:, 2] = np.array([[0., -1j], [1j, 0.]], dtype=complex)
    # pauli[0:, 0:, 3] = np.array([[1., 0.], [0., -1.]], dtype=complex)

    for i in range(0, 4 ** npart - 1):
        # there is 4 ** npart -1 operators in general,
        ope = np.array([[1]], dtype=complex)
        oi = i
        for ii in range(0, npart):
            flag = int(oi % 4)
            oi = (oi - flag)/4
            pauli_flag = pauli[0:, 0:, flag]
            ope = np.kron(ope, pauli_flag)
            # so that the operators[0:, 0:, 0] is identity and we will not use the first one
        operators[0:, 0:, i] = ope
    p_operators = np.ones([4 ** npart - 1], dtype=np.float32)
    p_operators = p_operators/(4 ** npart - 1)
    return operators


def operator_generator_with_p(npart, dim):
    # generate measuring operators, npart is the number of particles and dim is
    # the total Hilbert space dimension.
    operators = np.zeros((dim, dim, 4 ** npart), dtype=complex)
    # allocate space
    pauli = np.zeros((2, 2, 4), dtype=complex)
    pauli[0:, 0:, 3] = np.array([[1., 0.], [0., 1.]], dtype=complex)
    pauli[0:, 0:, 1] = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    pauli[0:, 0:, 2] = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
    pauli[0:, 0:, 0] = np.array([[1., 0.], [0., 0.]], dtype=complex)
    # generate basic pauli measurement operators
    # pauli[0:, 0:, 0] = np.array([[1., 0.], [0., 1.]], dtype=complex)
    # pauli[0:, 0:, 1] = np.array([[0., 1.], [1., 0.]], dtype=complex)
    # pauli[0:, 0:, 2] = np.array([[0., -1j], [1j, 0.]], dtype=complex)
    # pauli[0:, 0:, 3] = np.array([[1., 0.], [0., -1.]], dtype=complex)

    for i in range(0, 4 ** npart - 1):
        # there is 4 ** npart -1 operators in general,
        ope = np.array([[1]], dtype=complex)
        oi = i
        for ii in range(0, npart):
            flag = int(oi % 4)
            oi = (oi - flag)/4
            pauli_flag = pauli[0:, 0:, flag]
            ope = np.kron(ope, pauli_flag)
            # so that the operators[0:, 0:, 0] is identity and we will not use the first one
        operators[0:, 0:, i] = ope
    p_operators = np.ones([4 ** npart - 1], dtype=np.float32)
    p_operators = p_operators/(4 ** npart - 1)
    return operators, p_operators


def get_alps(r, dim):
    """
    ad = int(dim*(dim+1)/2)
    alps = np.zeros([ad], dtype=complex)
    for i in range(0, dim):
        alps[int(((i+1)**2-i-1)/2):int(((i+1)**2+i+1)/2):1] = r[0:i+1:1, i]
    return alps
    # """
    alps = np.zeros([dim ** 2+dim], dtype=np.float32)
    count = 0
    for i in range(dim):
        alps[count:count + i + 1:1] = r.real[0:i + 1:1, i]
        count += i + 1
    for i in range(dim):
        alps[count:count + i + 1:1] = r.imag[0:i + 1:1, i]
        count += i + 1
    return alps


def get_alps_label(r, dim):
    alps = np.zeros([dim**2], dtype=np.float32)
    count = 0
    for i in range(dim):
        alps[count:count+i+1:1] = r.real[0:i+1:1, i]
        count += i+1
    for i in range(1, dim):
        alps[count:count+i:1] = r.imag[0:i:1, i]
        count += i
    return alps


def rebuild_r(alps, dim):
    """
    r = np.zeros([dim, dim], dtype=complex)
    for i in range(0, dim):
        r[0:i+1:1, i] = alps[int(((i+1)**2-i-1)/2):int(((i+1)**2+i+1)/2):1]
    return r
    # """
    # """
    r = np.zeros([dim, dim], dtype=complex)
    count = 0
    for i in range(dim):
        r[0:i + 1:1, i] = alps[count:count + i + 1:1]
        count += i + 1
    for i in range(dim):
        r[0:i + 1:1, i] += 1j * alps[count:count + i + 1:1]
        count += i + 1
    return r
    # """


def rebuild_r_label(alps, dim):
    r = np.zeros([dim, dim], dtype=complex)
    count = 0
    for i in range(dim):
        r[0:i+1:1, i] = alps[count:count+i+1:1]
        count += i+1
    for i in range(1, dim):
        r[0:i:1, i] += 1j * alps[count:count+i:1]
        count += i
    return r


def r_to_rho(r):
    rho = np.matmul(Sf.hermite(r), r)
    tr = np.trace(rho).real
    rho = rho/tr
    return rho, tr


def noise_wn(rho, npart, dim, mode, par1):
    whitenoise = np.eye(dim)/dim
    rhon = np.zeros([dim, dim], dtype=complex)  # allocate space for rho with noise
    if mode == 0:
        rhon = (1-par1) * rho + par1 * whitenoise
        # depolarization noise, r' = (1 - p)r + p * white noise
    if mode == 1:
        rn = random.random(size=(1, 1))
        if rn <= par1:
            rhon = whitenoise
        if rn > par1:
            rhon = rho
    return rhon


def noise_bf(rho, npart, dim, mode, par2):
    id = np.eye(2)
    bf = Sf.pauli(1)
    n_rho = np.zeros([dim, dim], dtype=complex)
    if mode == 0:
        for ind in range(2**npart):
            tempo = np.array([[1]], dtype=complex)
            tempp = 1.
            for n in range(npart):
                if ind & (2**n) != 0:
                    tempo = np.kron(tempo, bf)
                    tempp = tempp * par2[n]
                else:
                    tempo = np.kron(tempo, id)
                    tempp = tempp * (1 - par2[n])
            n_rho = n_rho + tempp * np.dot(np.dot(tempo, rho), tempo)
    if mode == 1:
        nps = random.random(size=[npart])
        tempo = np.array([[1]])
        for n in range(npart):
            if nps[n] < par2[n]:
                tempo = np.kron(tempo, bf)
            else:
                tempo = np.kron(tempo, id)
        n_rho = np.dot(np.dot(tempo, rho), tempo)
    return n_rho


def noise_bf_pf(rho, npart, dim, mode, par2, par3):
    id = np.eye(2)
    bf = Sf.pauli(1)
    pf = Sf.pauli(3)
    n_rho = np.zeros([dim, dim], dtype=complex)
    if mode == 0 or mode == 1:
        for ind in range(4**npart):
            tempo = np.array([[1]], dtype=complex)
            tempp = 1.
            for n in range(npart):
                tempo2 = np.eye(2)
                if ind & (2**(n*2)) != 0:
                    tempo2 = np.dot(tempo2, bf)
                    tempp = tempp * par2[n]
                else:
                    tempo2 = np.dot(tempo2, id)
                    tempp = tempp * (1 - par2[n])
                if ind & (2**(n*2)*2) != 0:
                    tempo2 = np.dot(tempo2, pf)
                    tempp = tempp * par3[n]
                else:
                    tempo2 = np.dot(tempo2, id)
                    tempp = tempp * (1 - par3[n])
                tempo = np.kron(tempo, tempo2)
            n_rho = n_rho + tempp * np.dot(np.dot(tempo, rho), tempo)
    '''
    if mode == 1:
        nps = random.random(size=[npart])
        tempo = np.array([[1]])
        for n in range(npart):
            if nps[n] < par2[n]:
                tempo = np.kron(tempo, bf)
            else:
                tempo = np.kron(tempo, id)
        n_rho = np.dot(np.dot(tempo, rho), tempo)
    '''
    return n_rho


def noise_ad(rho, npart, dim, mode, par):
    e0 = np.zeros([2, 2, npart], dtype=complex)
    e1 = np.zeros([2, 2, npart], dtype=complex)
    for i in range(npart):
        e0[:, :, i] = np.array([[1., 0], [0., np.sqrt(1-par[i])]])
        e1[:, :, i] = np.array([[0., np.sqrt(par[i])], [0., 0.]])
    n_rho = np.zeros([dim, dim], dtype=complex)
    if mode == 0:
        for ind in range(2**npart):
            tempo = np.array([[1.]])
            for n in range(npart):
                if ind & 2**n == 0:
                    tempo = np.kron(tempo, e0[:, :, n])
                else:
                    tempo = np.kron(tempo, e1[:, :, n])
            n_rho = n_rho + np.dot(np.dot(tempo, rho), tempo.T)
    if mode == 1:
        for ind in range(2**npart):
            tempo = np.array([[1.]])
            for n in range(npart):
                if ind & 2**n == 0:
                    tempo = np.kron(tempo, e0[:, :, n])
                else:
                    tempo = np.kron(tempo, e1[:, :, n])
            n_rho = n_rho + np.dot(np.dot(tempo, rho), tempo.T)
    return n_rho


def measure(npart, dim, operators, rhot, measure_times, mode, par1, par2, par3, par4):
    frequencies = np.zeros([4**npart-1], dtype=np.float32)
    if mode == 0:
        for i in range(0, 4**npart-1):
            m = operators[0:, 0:, i+1]
            # the reason of i+1, see above in operator_generator
            # rhom = noise_bf(rhot, npart, dim, mode, par2)
            rhom = noise_ad(rhot, npart, dim, mode, par4)
            rhom = noise_bf_pf(rhom, npart, dim, mode, par2, par3)
            rhom = noise_wn(rhom, npart, dim, mode, par1)
            # get the state after noise
            frequencies[i] = np.real(np.trace(np.dot(m, rhom)))
            # f=tr(M*rho)
    if mode == 1:
        for i in range(0, 4**npart-1):
            count = 0.
            m = operators[0:, 0:, i+1]
            rhom = noise_ad(rhot, npart, dim, 0, par4)
            rhom = noise_bf_pf(rhom, npart, dim, 0, par2, par3)
            rhom = noise_wn(rhom, npart, dim, 0, par1)
            mn = np.real(np.trace(np.dot(m, rhom)))
            rns = random.random(size=(measure_times, 1))
            for j in range(measure_times):
                rn = rns[j]
                if rn <= mn:
                    count += 1
            frequencies[i] = count/measure_times
    return frequencies


def noise_old(rho, npart, dim, mode, par1):

    # write the originally Pauli operators
    x0 = np.array([[0, 1], [1, 0]], dtype=complex)
    y0 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z0 = np.array([[1, 0], [0, -1]], dtype=complex)

    # initialize
    x = np.array([[1]], dtype=complex)
    y = np.array([[1]], dtype=complex)
    z = np.array([[1]], dtype=complex)
    whitenoise = np.eye(dim)/dim

    rhon = np.zeros([dim, dim], dtype=complex)  # allocate space for rho with noise
    for i in range(0, npart):
        x = np.kron(x, x0)
        y = np.kron(y, y0)
        z = np.kron(z, z0)
    if mode == 0:
        # rhon = (1-par1) * rho + par1/3*(np.dot(np.dot(x, rho), x)+np.dot(np.dot(y, rho), y)+np.dot(np.dot(z, rho), z))
        rhon = (1-par1) * rho + par1 * whitenoise
        # depolarization noise, r'=(1-p)r+p/3(XrX+YrY+ZrZ)
    if mode == 1:
        rn = random.random(size=(1, 1))
        if rn <= par1/3:
            rhon = np.dot(np.dot(x, rho), x)
        if par1/3 < rn <= par1*2/3:
            rhon = np.dot(np.dot(y, rho), y)
        if par1*2/3 < rn <= par1:
            rhon = np.dot(np.dot(z, rho), z)
        if rn > par1:
            rhon = rho
    return rhon
