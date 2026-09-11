"""
EvalFunctions.py
================
Evaluation helpers for neural-network quantum state tomography.

This module is a verbatim extraction of `performance()` from main.py, kept as a
separate file for code organisation (2026-09-10 split). No logic was changed;
all comments and commented-out alternative paths are preserved exactly so that
"temporarily unused" or "backup" code paths remain available.

Public API
----------
performance(output_array, dimension, number, rhos)
    Reconstruct density matrices from network outputs, compute fidelity vs.
    ground truth, and return per-sample fidelity + reconstructed rhos +
    inferred noise parameters.
"""

import numpy as np
import NoiseDetectFunctions as Ndf
import SupportFunctions as Sf


def performance(output_array, dimension, number, rhos):
    # this function is to evaluate the performance of the network
    # output is tensor format, from tensorflow's network
    # dimension is the dimension of the density matrix, d*d
    # number is the total number of the states,
    # rhos is the origin states that put into test
    # output_array = np.array(output)  # turn the tensor into ndarray
    fidelity = np.zeros([number], dtype=np.float32)  # allocate space for fidelity
    noiseparas = output_array[:, dimension**2:-1]
    id = np.eye(dimension) / dimension

    # density matrix
    # '''
    re_rhos = np.zeros([dimension, dimension, number], dtype=complex)  # allocate space for reconstructed rhos
    for j in range(number):
        alps = output_array[j, 0:].T  # get the output label
        # alps = alps_pre[0:para_number] + 1j * alps_pre[para_number:2*para_number]  # build the alpha vector
        r = Ndf.rebuild_r_label(alps, dimension)  # build R (after Cholesky decompose)
        [re_rho, tr] = Ndf.r_to_rho(r)  # build rho = R'*R
        noiseparas[j, :] = alps[dimension**2:-1]/tr
        # p = alps[dimension**2]
        re_rhos[0:, 0:, j] = re_rho  # save rebuild rho
        rho = rhos[0:, 0:, j]  # get the original rho
        fidelity[j] = Sf.fidelity(re_rho, rho)  # calculate fidelity
        # psi = Sf.ghz_state(5)
        # fidelity[j] = Sf.fidelity_pure(psi, re_rho)
    # '''

    # vector
    '''
    # re_rhos = np.zeros([number, dimension], dtype=complex)
    re_rhos = output_array[:, 0:dimension] + 1j * output_array[:, dimension:dimension*2]
    for j in range(number):
        rho_normalized = Sf.normalizing_vector(re_rhos[j, :])
        fidelity[j] = Sf.fidelity_vectors(rhos[j, :], rho_normalized)
    # '''

    mean_fidelity = np.mean(fidelity)  # get mean fidelity
    return fidelity, mean_fidelity, re_rhos, noiseparas
