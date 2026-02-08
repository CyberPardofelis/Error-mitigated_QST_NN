import numpy as np
from numpy import random
from scipy import linalg
from scipy.linalg import sqrtm
import math


def random_density_matrix(dim):
    # simplified version of the function with the same name in QETLAB 0.9
    ar = random.random(size=(dim, dim))
    ai = random.random(size=(dim, dim))
    ar = 2 * ar - 1
    ai = 2 * ai - 1
    a = ar + 1j * ai
    at = ar.T - 1j * ai.T
    # it's not very easy for numpy to get a Hermite conjugate,
    # luckily we have the real part and the imaginary part of the matrix
    rho = np.dot(a, at)
    rho = rho/np.trace(rho)
    return rho


def hermite(a):
    ac = a - 2j*a.imag  # conjugate
    ac = ac.T  # transpose
    return ac


def chol(a):
    # just to reduce the words that need to type
    r = linalg.cholesky(a)
    return r


def fidelity(r1, r2):
    f = np.trace(sqrtm(np.dot(np.dot(sqrtm(r1), r2), sqrtm(r1))))
    # f=tr(sqrt(sqrt(r1)*r2*sqrt(r2)))
    return f.real


def fidelity_pure(psi, rho):
    f=np.sqrt(np.dot(psi.T.conj(), np.dot(rho, psi)))
    return f.real


def pauli(n):
    # Pauli operators, 0=identity, 1=sigma_x, 2=sigma_y, 3=sigma_z
    if n == 0:
        p = np.array([[1, 0], [0, 1]], dtype=complex)
    if n == 1:
        p = np.array([[0, 1], [1, 0]], dtype=complex)
    if n == 2:
        p = np.array([[0, -1j], [1j, 0]], dtype=complex)
    if n == 3:
        p = np.array([[1, 0], [0, -1]], dtype=complex)
    return p


def get_alps_label(r, dim):
    # reshape the upper triangle matrix into a vector, also separate the real part and imaginary part.
    # so that the information of a density matrix could be put into a real vector.
    alps = np.zeros([dim**2], dtype=np.float32)
    count = 0
    for i in range(dim):
        alps[count:count+i+1:1] = r.real[0:i+1:1, i]
        count += i+1
    for i in range(1, dim):
        alps[count:count+i:1] = r.imag[0:i:1, i]
        count += i
    return alps


def rebuild_r_label(alps, dim):
    # reshape the vector above into the upper triangle matrix format.
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
    # from the upper triangle matrix format to the original density matrix.
    # the additional freedom is for normalization
    rho = np.matmul(hermite(r), r)
    tr = np.trace(rho).real
    rho = rho/tr
    return rho, tr


def n_choose_k(n, k):
    tabel = []
    for i in range(2**n):
        temp = list(map(int, bin(i)[2:]))  # 将十进制数转为二进制数并以数组形式存储
        temp_total = np.zeros([n, 1], dtype=int)  # 上面不会存成指定长度的样子
        length = len(temp)  # 这一段用来把不到整个长度的temp数组丢进temp_total的对应位置
        for j in range(length):
            temp_total[n-length+j] = [temp[j]]
        ones = 0
        for j in range(n):  # 数1的数量 刚好为k的保留
            if temp_total[j] == 1:
                ones += 1
        if ones == k:
            tabel.append(temp_total)
            # 需要注意 这个tabel在调用的时候 第一层选择第几个数组 往后再是另一层 它不是一个矩阵
    return tabel


def dicke_state(n, k):
    tabel = n_choose_k(n, k)  # 找出叠加态的所有态矢
    ket0 = np.array([[1], [0]])  # 定义0ket和1ket
    ket1 = np.array([[0], [1]])
    state = np.zeros([2**n, 1])  # 求和器
    cnk = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))  # 计算组合数 遍历table以及归一化用
    for i in range(int(cnk)):  # 遍历table
        temp = np.array([[1]])  # 乘法器
        for j in range(n):  # 计算张量积
            if tabel[i][j] == 0:
                temp = np.kron(temp, ket0)
            if tabel[i][j] == 1:
                temp = np.kron(temp, ket1)
        state = state + temp  # 态叠加
    state = state / np.sqrt(cnk)  # 归一化
    return state


def ghz_state(npart):
    # GHZ
    r = np.zeros([2**npart, 1])
    r[0] = np.sqrt(2) / 2
    r[-1] = np.sqrt(2) / 2
    return r


def w_state(npart):
    # W
    r = np.zeros([2**npart, 1])
    for i in range(npart):
        r[2**i] = 1/np.sqrt(npart)
    return r


def random_pure_state(npart):
    rr = np.random.random([2**npart, 1])
    ri = np.random.random([2**npart, 1])
    r = rr + 1j * ri
    r2 = np.dot(np.transpose(np.conj(r)), r)
    r = r/np.sqrt(r2)
    return r


def ghz_like_state(npart, theta):
    r = np.zeros([2 ** npart, 1])
    r[0] = np.sin(theta)
    r[-1] = np.cos(theta)
    return r


def fidelity_vectors(psi1, psi2):
    ip = np.sum(psi1*psi2.conj())
    return np.sqrt(ip).real

def normalizing_vector(psi):
    ip = np.sum(psi*psi.conj())
    psi = psi/np.sqrt(ip)
    return psi