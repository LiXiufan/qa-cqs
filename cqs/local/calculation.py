########################################################################################################################
# Copyright (c) Xiufan Li. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Xiufan Li
# Supervisor: Patrick Rebentrost
# Institution: Centre for Quantum Technologies, National University of Singapore
# For feedback, please contact Xiufan at: shenlongtianwu8@gmail.com.
########################################################################################################################


# !/usr/bin/env python3

"""
    This is the calculator module to compute Q and r according to the Hadamard test shooting outcomes.
"""

from numpy import array
from numpy import zeros, identity
from numpy import append

from numpy import real, imag
from numpy import conj

from hardware.execute import Hadamard_test


# from qiskit_ionq import IonQProvider
# from qiskit_braket_provider import AWSBraketProvider
# import time
# from datetime import datetime

# BRAKET_DEVICE = 'SV1'
# BRAKET_DEVICE = 'Aria 1'
# BRAKET_DEVICE = 'Harmony'

def U_list_dagger(U):
    return U[::-1]


def __submit_all_inner_products_in_V_dagger_V(instance, ansatz_tree, **kwargs):
    r"""
        Estimate all independent inner products that appear in matrix `V_dagger_V`.
        Note that we only estimate all upper triangular elements and all diagonal elements.
        For lower elements, since the matrix `V_dagger_V` is symmetrical, we fill in the elements using
        dagger of each estimated inner product. Thus the symmetry is maintained.
        In total, we are going to estimate
            total number of elements = 1/2 * (`tree_depth` + 1) * `tree_depth`
    """

    unitaries = instance.get_unitaries()
    n = instance.get_num_qubit()
    num_term = instance.get_num_term()
    Ub = instance.get_ub()
    tree_depth = len(ansatz_tree)
    ip_idxes = [[0 for _ in range(tree_depth)] for _ in range(tree_depth)]
    for i in range(tree_depth):
        for j in range(i, tree_depth):
            element_idxes = [[0 for _ in range(num_term)] for _ in range(num_term)]
            for k in range(num_term):
                for l in range(num_term):
                    U1 = unitaries[k].compose(ansatz_tree[i], qubits=unitaries[k].qubits)
                    U2 = unitaries[l].compose(ansatz_tree[j], qubits=unitaries[l].qubits)
                    inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
                    element_idxes[k][l] = inner_product
            ip_idxes[i][j] = element_idxes
    return ip_idxes


def __submit_all_inner_products_in_q(instance, ansatz_tree, **kwargs):
    r"""
        Estimate all independent inner products that appear in vector `q`
    """

    unitaries = instance.get_unitaries()
    n = instance.get_num_qubit()
    num_term = instance.get_num_term()
    Ub = instance.get_ub()
    tree_depth = len(ansatz_tree)
    ip_idxes = [0 for _ in range(tree_depth)]
    for i in range(tree_depth):
        element_idxes = [0 for _ in range(num_term)]
        for k in range(num_term):
            U1 = ansatz_tree[i]
            U2 = unitaries[k]
            inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
            element_idxes[k] = inner_product
        ip_idxes[i] = element_idxes
    return ip_idxes


def __retrieve__all_inner_products_in_V_dagger_V(instance, ansatz_tree, ip_idxes, backend='eigens'):
    r"""
        Retrieve the results of all submitted tasks.
    """
    if backend in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ip_idxes
    else:  # hardware retrieval
        return 0


def __retrieve__all_inner_products_in_q(instance, ansatz_tree, ip_idxes, backend='eigens'):
    r"""
        Retrieve the results of all submitted tasks.
    """
    if backend in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ip_idxes
    else:  # hardware retrieval
        return 0


def __estimate_V_dagger_V(instance, ansatz_tree, loss_type=None, backend=None, **kwargs):
    r"""
        Estimate all independent inner products that appear in matrix `V_dagger_V`.
        Note that we only estimate all upper triangular elements and all diagonal elements.
        For lower elements, since the matrix `V_dagger_V` is symmetrical, we fill in the elements using
        dagger of each estimated inner product. Thus the symmetry is maintained.
        In total, we are going to estimate
            total number of elements = 1/2 * (`tree_depth` + 1) * `tree_depth`
    """
    num_term = instance.get_num_term()
    coeffs = instance.get_coeffs()
    tree_depth = len(ansatz_tree)
    V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')

    ip_idxes = __submit_all_inner_products_in_V_dagger_V(instance, ansatz_tree, backend=backend, **kwargs)
    ip_values = __retrieve__all_inner_products_in_V_dagger_V(instance, ansatz_tree, ip_idxes, backend=backend)

    for i in range(tree_depth):
        for j in range(i, tree_depth):
            element_values = ip_values[i][j]
            item = 0
            for k in range(num_term):
                for l in range(num_term):
                    inner_product = element_values[k][l]
                    item += conj(coeffs[k]) * coeffs[l] * inner_product
            V_dagger_V[i][j] = item
            if i < j:
                V_dagger_V[j][i] = conj(item)
    if loss_type == 'l2reg':
        V_dagger_V += 1 / 2 * identity(tree_depth, dtype='complex128')
    return V_dagger_V


def __estimate_q(instance, ansatz_tree, backend=None, **kwargs):
    r"""
        Estimate all independent inner products that appear in vector `q`.
        In total, we are going to estimate
            total number of elements = `tree_depth`
    """

    num_term = instance.get_num_term()
    coeffs = instance.get_coeffs()
    tree_depth = len(ansatz_tree)
    q = zeros((tree_depth, 1), dtype='complex128')

    ip_idxes = __submit_all_inner_products_in_q(instance, ansatz_tree, backend=backend, **kwargs)
    ip_values = __retrieve__all_inner_products_in_q(instance, ansatz_tree, ip_idxes, backend=backend)

    for i in range(tree_depth):
        element_values = ip_values[i]
        item = 0
        for k in range(num_term):
            inner_product = element_values[k]
            item += conj(coeffs[k]) * inner_product
        q[i][0] = item
    return q


def __reshape_to_Q_r(matrix, vector):
    Vr = real(matrix)
    Vi = imag(matrix)
    # Q = R  - I
    #     I   R
    Q = array(append(append(Vr, -Vi, axis=1), append(Vi, Vr, axis=1), axis=0), dtype='float64')

    qr = real(vector)
    qi = imag(vector)
    # r = Re(q)
    #     Im(q)
    r = array(append(qr, qi, axis=0), dtype='float64')
    return Q, r


def calculate_Q_r(instance, ansatz_tree, loss_type=None, **kwargs):
    r"""
        Please note that the objective function of CVXOPT has the form:   1/2  x^T P x  +   q^T x
        But our objective function is:                                         z^T Q z  - 2 r^T z + 1
        So the coefficients are corrected here.

    :param R:
    :param I:
    :param q:
    :return: Q, r
    """
    V_dagger_V = __estimate_V_dagger_V(instance, ansatz_tree, loss_type=loss_type, **kwargs)
    q = __estimate_q(instance, ansatz_tree, **kwargs)
    Q, r = __reshape_to_Q_r(V_dagger_V, q)
    return Q, r





# def submit_tasks_braket(instance, ansatz_tree, shot_frugal=None):
#     if shot_frugal is None:
#         shots = 1000
#     coeffs = instance.get_coeffs()
#     unitaries = instance.get_unitaries()
#     num_term = instance.get_num_term()
#     tree_depth = len(ansatz_tree)
#     V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')
#
#     provider = AWSBraketProvider()
#     simulator_backend = provider.get_backend(BRAKET_DEVICE)
#
#     Job_ids_K_R = []
#     Job_ids_K_I = []
#     Job_ids_q_R = []
#     Job_ids_q_I = []
#
#     for i in range(tree_depth):
#         for j in range(tree_depth):
#             for k in range(num_term):
#                 for l in range(num_term):
#                     u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(unitaries[k]) + unitaries[l] + \
#                         ansatz_tree[j]
#                     jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     file1 = open(file_name, "a")
#                     file1.writelines(["jobid u_K, real:", str(jobid_R), '\n'])
#                     file1.writelines(["jobid u_K, imag:", str(jobid_I), '\n'])
#                     file1.close()
#                     Job_ids_K_R.append(jobid_R)
#                     Job_ids_K_I.append(jobid_I)
#
#     for i in range(tree_depth):
#         for k in range(A_terms_number):
#             u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#             shots = P_A[k]
#             shots = 400
#             jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                           tasks_num=tasks_num, shots_num=shots_num)
#             jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                           tasks_num=tasks_num, shots_num=shots_num)
#             file1 = open(file_name, "a")
#             file1.writelines(["jobid u_q, real:", str(jobid_R), '\n'])
#             file1.writelines(["jobid u_q, imag:", str(jobid_I), '\n'])
#             file1.close()
#             Job_ids_q_R.append(jobid_R)
#             Job_ids_q_I.append(jobid_I)
#
#     exp_K_R = calculate_statistics(simulator_backend, Job_ids_K_R, file_name=file_name)
#     exp_K_I = calculate_statistics(simulator_backend, Job_ids_K_I, file_name=file_name)
#     exp_q_R = calculate_statistics(simulator_backend, Job_ids_q_R, file_name=file_name)
#     exp_q_I = calculate_statistics(simulator_backend, Job_ids_q_I, file_name=file_name)
#
#     Job_ids_regterm_R = []
#     Job_ids_regterm_I = []
#
#     if loss_type == 'l2reg':
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 u_reg = U_list_dagger(ansatz_tree[i]) + ansatz_tree[j]
#                 shots = 400
#                 jobid_R, tasks_num, shots_num = Hadamard_test(u_reg, backend=backend, alpha=1, shots=shots,
#                                                               tasks_num=tasks_num, shots_num=shots_num)
#                 jobid_I, tasks_num, shots_num = Hadamard_test(u_reg, backend=backend, alpha=1j, shots=shots,
#                                                               tasks_num=tasks_num, shots_num=shots_num)
#                 file1 = open(file_name, "a")
#                 file1.writelines(["jobid u_reg, real:", str(jobid_R), '\n'])
#                 file1.writelines(["jobid u_reg, imag:", str(jobid_I), '\n'])
#                 file1.close()
#                 Job_ids_regterm_R.append(jobid_R)
#                 Job_ids_regterm_I.append(jobid_I)
#         exp_reg_R = calculate_statistics(simulator_backend, Job_ids_regterm_R, file_name=file_name)
#         exp_reg_I = calculate_statistics(simulator_backend, Job_ids_regterm_I, file_name=file_name)
#
#     if loss_type == 'hamiltonian':
#         for i in range(tree_depth):
#             for k in range(A_terms_number):
#                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#                 shots = P_A[k]
#                 shots = 400
#                 jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                               tasks_num=tasks_num, shots_num=shots_num)
#                 jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                               tasks_num=tasks_num, shots_num=shots_num)
#                 file1 = open(file_name, "a")
#                 file1.writelines(["jobid u_q, real:", str(jobid_R), '\n'])
#                 file1.writelines(["jobid u_q, imag:", str(jobid_I), '\n'])
#                 file1.close()
#                 Job_ids_q_R.append(jobid_R)
#                 Job_ids_q_I.append(jobid_I)
#
#     for i in range(tree_depth):
#         for j in range(tree_depth):
#             # Uniform distribution of the shots
#             item = 0
#             for k in range(A_terms_number):
#                 for l in range(A_terms_number):
#                     inner_product_real = exp_K_R[i * tree_depth * A_terms_number * A_terms_number +
#                                                  j * A_terms_number * A_terms_number +
#                                                  k * A_terms_number +
#                                                  l]
#                     inner_product_imag = exp_K_I[i * tree_depth * A_terms_number * A_terms_number +
#                                                  j * A_terms_number * A_terms_number +
#                                                  k * A_terms_number +
#                                                  l]
#                     inner_product = inner_product_real - inner_product_imag * 1j
#                     item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#             if loss_type == 'l2reg':
#                 inner_product_real = exp_reg_R[i * tree_depth + j]
#                 inner_product_imag = exp_reg_I[i * tree_depth + j]
#                 inner_product = inner_product_real - inner_product_imag * 1j
#                 item += 0.5 * inner_product
#             V_dagger_V[i][j] = item
#
#     R = real(V_dagger_V)
#     I = imag(V_dagger_V)
#
#     q = zeros((tree_depth, 1), dtype='complex128')
#     for i in range(tree_depth):
#         item = 0
#         for k in range(A_terms_number):
#             inner_product_real = exp_q_R[i * A_terms_number + k]
#             inner_product_imag = exp_q_I[i * A_terms_number + k]
#             inner_product = inner_product_real - inner_product_imag * 1j
#             item += conj(A_coeffs[k]) * inner_product
#         q[i][0] = item
#
#
#     id_list = []
#
#     return id_list


# def construct_auxiliary_system():


# def calculate_Q_r_by_Hadamrd_test(instance, ansatz_tree,
#                                   loss_type=None, backend=None,
#                                   shots_budget=1024, frugal=False, file_name='message.txt'):
#     """
#         Please note that the objective function of CVXOPT has the form:   1/2  x^T P x  +   q^T x
#         But our objective function is:                                         z^T Q z  - 2 r^T z + 1
#         So the coefficients are corrected here.
#
#     :param R:
#     :param I:
#     :param q:
#     :return: Q, r
#     """
#
#     coeffs = instance.get_coeffs()
#     unitaries = instance.get_unitaries()
#     ub = instance.get_ub()
#     num_term = instance.get_num_term()
#     tree_depth = len(ansatz_tree)
#     V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')
#     if loss_type is None:
#         loss_type = 'l2'
#     if backend is None:
#         backend = 'eigens'
#     # if frugal is True:
#     #     shots_each_entry = shots_budget / (10 * (tree_depth ** 2 + tree_depth))
#     #     M_A_A = sum(abs(conj(A_coeffs[k]) * A_coeffs[l])
#     #                 for k in range(A_terms_number) for l in range(A_terms_number))
#     #     P_A_A = [10 * int(shots_each_entry * (abs(conj(A_coeffs[k]) * A_coeffs[l]) / M_A_A))
#     #              for k in range(A_terms_number) for l in range(A_terms_number)]
#     #     M_A = sum([abs(conj(A_coeffs[k]))
#     #                for k in range(A_terms_number)])
#     #     P_A = [10 * int(shots_each_entry * (abs(conj(A_coeffs[k])) / M_A)) for k in range(A_terms_number)]
#     #
#     # else:
#     #     P_A_A = [100 for _ in range(A_terms_number) for _ in range(A_terms_number)]
#     #     P_A = [100 for _ in range(A_terms_number)]
#
#
#
#     if backend == 'braket':
#         provider = AWSBraketProvider()
#         simulator_backend = provider.get_backend(BRAKET_DEVICE)
#         Job_ids_K_R = []
#         Job_ids_K_I = []
#         Job_ids_q_R = []
#         Job_ids_q_I = []
#
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 # Uniform distribution of the shots
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
#                             ansatz_tree[j]
#                         shots = P_A_A[k * A_terms_number + l]
#                         shots = 400
#                         jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                       tasks_num=tasks_num, shots_num=shots_num)
#                         jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                       tasks_num=tasks_num, shots_num=shots_num)
#                         file1 = open(file_name, "a")
#                         file1.writelines(["jobid u_K, real:", str(jobid_R), '\n'])
#                         file1.writelines(["jobid u_K, imag:", str(jobid_I), '\n'])
#                         file1.close()
#                         Job_ids_K_R.append(jobid_R)
#                         Job_ids_K_I.append(jobid_I)
#
#         for i in range(tree_depth):
#             for k in range(A_terms_number):
#                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#                 shots = P_A[k]
#                 shots = 400
#                 jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                               tasks_num=tasks_num, shots_num=shots_num)
#                 jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                               tasks_num=tasks_num, shots_num=shots_num)
#                 file1 = open(file_name, "a")
#                 file1.writelines(["jobid u_q, real:", str(jobid_R), '\n'])
#                 file1.writelines(["jobid u_q, imag:", str(jobid_I), '\n'])
#                 file1.close()
#                 Job_ids_q_R.append(jobid_R)
#                 Job_ids_q_I.append(jobid_I)
#
#         exp_K_R = calculate_statistics(simulator_backend, Job_ids_K_R, file_name=file_name)
#         exp_K_I = calculate_statistics(simulator_backend, Job_ids_K_I, file_name=file_name)
#         exp_q_R = calculate_statistics(simulator_backend, Job_ids_q_R, file_name=file_name)
#         exp_q_I = calculate_statistics(simulator_backend, Job_ids_q_I, file_name=file_name)
#
#         Job_ids_regterm_R = []
#         Job_ids_regterm_I = []
#
#         if loss_type == 'l2reg':
#             for i in range(tree_depth):
#                 for j in range(tree_depth):
#                     u_reg = U_list_dagger(ansatz_tree[i]) + ansatz_tree[j]
#                     shots = 400
#                     jobid_R, tasks_num, shots_num = Hadamard_test(u_reg, backend=backend, alpha=1, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     jobid_I, tasks_num, shots_num = Hadamard_test(u_reg, backend=backend, alpha=1j, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     file1 = open(file_name, "a")
#                     file1.writelines(["jobid u_reg, real:", str(jobid_R), '\n'])
#                     file1.writelines(["jobid u_reg, imag:", str(jobid_I), '\n'])
#                     file1.close()
#                     Job_ids_regterm_R.append(jobid_R)
#                     Job_ids_regterm_I.append(jobid_I)
#             exp_reg_R = calculate_statistics(simulator_backend, Job_ids_regterm_R, file_name=file_name)
#             exp_reg_I = calculate_statistics(simulator_backend, Job_ids_regterm_I, file_name=file_name)
#
#         if loss_type == 'hamiltonian':
#             for i in range(tree_depth):
#                 for k in range(A_terms_number):
#                     u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#                     shots = P_A[k]
#                     shots = 400
#                     jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     file1 = open(file_name, "a")
#                     file1.writelines(["jobid u_q, real:", str(jobid_R), '\n'])
#                     file1.writelines(["jobid u_q, imag:", str(jobid_I), '\n'])
#                     file1.close()
#                     Job_ids_q_R.append(jobid_R)
#                     Job_ids_q_I.append(jobid_I)
#
#
#
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 # Uniform distribution of the shots
#                 item = 0
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         inner_product_real = exp_K_R[i * tree_depth * A_terms_number * A_terms_number +
#                                                      j * A_terms_number * A_terms_number +
#                                                      k * A_terms_number +
#                                                      l]
#                         inner_product_imag = exp_K_I[i * tree_depth * A_terms_number * A_terms_number +
#                                                      j * A_terms_number * A_terms_number +
#                                                      k * A_terms_number +
#                                                      l]
#                         inner_product = inner_product_real - inner_product_imag * 1j
#                         item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#                 if loss_type == 'l2reg':
#                     inner_product_real = exp_reg_R[i * tree_depth + j]
#                     inner_product_imag = exp_reg_I[i * tree_depth + j]
#                     inner_product = inner_product_real - inner_product_imag * 1j
#                     item += 0.5 * inner_product
#                 V_dagger_V[i][j] = item
#
#         R = real(V_dagger_V)
#         I = imag(V_dagger_V)
#
#         q = zeros((tree_depth, 1), dtype='complex128')
#         for i in range(tree_depth):
#             item = 0
#             for k in range(A_terms_number):
#                 inner_product_real = exp_q_R[i * A_terms_number + k]
#                 inner_product_imag = exp_q_I[i * A_terms_number + k]
#                 inner_product = inner_product_real - inner_product_imag * 1j
#                 item += conj(A_coeffs[k]) * inner_product
#             q[i][0] = item
#
#     else:
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 item = 0
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
#                             ansatz_tree[j]
#                         shots = P_A_A[k * A_terms_number + l]
#                         shots = 400
#                         # file1 = open(file_name, "a")
#                         # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
#                         # file1.close()
#                         inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1,
#                                                                                  shots=shots, tasks_num=tasks_num,
#                                                                                  shots_num=shots_num)
#                         inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j,
#                                                                                  shots=shots, tasks_num=tasks_num,
#                                                                                  shots_num=shots_num)
#                         inner_product = inner_product_real - inner_product_imag * 1j
#                         item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#                 if loss_type == 'l2reg':
#                     u_reg = U_list_dagger(ansatz_tree[i]) + ansatz_tree[j]
#                     shots = 400
#                     inner_product_real, tasks_num, shots_num = Hadamard_test(u_reg, backend=backend, alpha=1, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     inner_product_imag, tasks_num, shots_num = Hadamard_test(u_reg, backend=backend, alpha=1j, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#
#                     inner_product = inner_product_real - inner_product_imag * 1j
#                     item += 0.5 * inner_product
#                 V_dagger_V[i][j] = item
#
#         R = real(V_dagger_V)
#         I = imag(V_dagger_V)
#
#         q = zeros((tree_depth, 1), dtype='complex128')
#         for i in range(tree_depth):
#             item = 0
#             for k in range(A_terms_number):
#                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#                 shots = P_A[k]
#                 shots = 400
#                 # file1 = open(file_name, "a")
#                 # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
#                 # file1.close()
#                 inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                          tasks_num=tasks_num, shots_num=shots_num)
#                 inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                          tasks_num=tasks_num, shots_num=shots_num)
#                 inner_product = inner_product_real - inner_product_imag * 1j
#                 item += conj(A_coeffs[k]) * inner_product
#             q[i][0] = item
#
#     # Q     =      R    -I
#     #       =      I     R
#     Q = array(append(append(R, -I, axis=1), append(I, R, axis=1), axis=0), dtype='float64')
#
#     # r = [Re(q),
#     #      Im(q)]
#     r_real = real(q)
#     r_imag = imag(q)
#     r = array(append(r_real, r_imag, axis=0), dtype='float64')
#     return Q, r, tasks_num, shots_num

########################################################################################################################
# IonQ Access
# def calculate_statistics(jobs):
#     # a list of jobs
#     exps = []
#     for job in jobs:
#         if '0' not in job.get_probabilities().keys():
#             p0 = 0
#             p1 = 1
#         elif '1' not in job.get_probabilities().keys():
#             p0 = 1
#             p1 = 0
#         else:
#             p0 = job.get_probabilities()['0']
#             p1 = job.get_probabilities()['1']
#             # p0 = 0.5
#             # p1 = 0.5
#         exp = p0 - p1
#         exps.append(exp)
#     return exps


# Braket Access
# def calculate_statistics(backend, jobs_ids, file_name='message.txt'):
#     # a list of jobs
#     exps = []
#     for job_id in jobs_ids:
#         if job_id == 1 or job_id == 0:
#             exp = job_id
#             # print("This circuit is composed of identities, skip.")
#             file1 = open(file_name, "a")
#             file1.writelines(["This circuit is composed of identities, skip.\n"])
#             file1.close()
#         else:
#             job = backend.retrieve_job(job_id)
#             # status = job.status()
#             # now = datetime.now()
#             # current_time = now.strftime("%H:%M:%S")
#             # file1 = open(file_name, "a")
#             # file1.writelines(["\nCurrent Time =", str(current_time), '\n'])
#             # file1.writelines(["Current Status:", str(status), '\n\n'])
#             # file1.close()
#             # print("Current Time =", current_time)
#             # print('Current Status:', status)
#             # print()
#             # DONE = status.DONE
#             # while status != DONE:
#             #     time.sleep(3600)
#             #     now = datetime.now()
#             #     current_time = now.strftime("%H:%M:%S")
#             #     status = job.status()
#             #     file1 = open(file_name, "a")
#             #     file1.writelines(["\nCurrent Time =", str(current_time), '\n'])
#             #     file1.writelines(["Current Status:", str(status), '\n\n'])
#             #     file1.close()
#             #     print("Current Time =", current_time)
#             #     print('Status:', status)
#             #     print()
#             # while not is_result_availble():
#             #     # block for a moment
#             #     sleep(1)
#             count = backend.retrieve_job(job_id).result().get_counts()
#             new_count = {'0': 0, '1': 0}
#             for k in count.keys():
#                 new_count[k[-1]] += count[k]
#             count = new_count
#             file1 = open(file_name, "a")
#             file1.writelines(["The sampling result is:", str(count), '\n'])
#             file1.close()
#             # print("The sampling result is:", count)
#             if count['0'] == 0:
#                 p0 = 0
#                 p1 = 1
#             elif count['1'] == 0:
#                 p0 = 1
#                 p1 = 0
#             else:
#                 shots = sum(list(count.values()))
#                 p0 = count['0'] / shots
#                 p1 = count['1'] / shots
#             file1 = open(file_name, "a")
#             file1.writelines(["The sampling probability of getting 0 is: p0 =", str(p0), '\n'])
#             file1.writelines(["The sampling probability of getting 1 is: p1 =", str(p1), '\n'])
#             file1.close()
#             # print("The sampling probability of getting 0 is: p0 =", p0)
#             # print("The sampling probability of getting 1 is: p1 =", p1)
#             if count['0'] != 0 and count['1'] != 0:
#                 if p0 < 0.2:
#                     p0 = 0
#                     p1 = 1
#                     file1 = open(file_name, "a")
#                     file1.writelines(["p0 < 0.2: set p0 = 0 and p1 = 1.", '\n'])
#                     file1.writelines(["Expectation value is -1.", '\n'])
#                     file1.close()
#                     # print("p0 < 0.2: set p0 = 0 and p1 = 1.")
#                     # print("Expectation value is -1.")
#                 else:
#                     if p1 < 0.2:
#                         p0 = 1
#                         p1 = 0
#                         file1 = open(file_name, "a")
#                         file1.writelines(["p0 > 0.8: set p0 = 1 and p1 = 0.", '\n'])
#                         file1.writelines(["Expectation value is 1.", '\n'])
#                         file1.close()
#                         # print("p0 > 0.8: set p0 = 1 and p1 = 0.")
#                         # print("Expectation value is 1.")
#                     else:
#                         p0 = 0.5
#                         p1 = 0.5
#                         file1 = open(file_name, "a")
#                         file1.writelines(["0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.", '\n'])
#                         file1.writelines(["Expectation value is 0.", '\n'])
#                         file1.close()
#                         # print("0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.")
#                         # print("Expectation value is 0.")
#             # print()
#             exp = p0 - p1
#         exps.append(exp)
#     return exps
