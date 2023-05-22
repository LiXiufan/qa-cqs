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

from numpy import identity
from numpy import array, linalg
from numpy import zeros
from numpy import sqrt, append

from numpy import real, imag
from numpy import kron, conj, transpose

from utils import print_progress
from cqs_module.mitigation import Richardson_extrapolate

# from hardware.Qibo.qibo_access import Hadamard_test
# from hardware.IonQ.ionq_access import Hadamard_test as Hadamard_test_ibmq
from hardware.Eigens.eigens_access import Hadamard_test as Hadamard_test_eigens
from hardware.IBMQ.ibmq_access import Hadamard_test as Hadamard_test_ibmq

def I_mat():
    return array([[1, 0], [0, 1]])


def X_mat():
    return array([[0, 1], [1, 0]])


def Y_mat():
    return array([[0, -1j], [1j, 0]])


def Z_mat():
    return array([[1, 0], [0, -1]])


def H_mat():
    return 1 / sqrt(2) * array([[1, 1], [1, -1]])


def zero_state():
    return array([1, 0]).reshape(2, 1)


def one_state():
    return array([0, 1]).reshape(2, 1)


def plus_state():
    return (1 / sqrt(2)) * array([1, 1]).reshape(2, 1)


def minus_state():
    return (1 / sqrt(2)) * array([1, -1]).reshape(2, 1)


# def merge_unitaries(U1, U2):
#     return

def gate_to_matrix(gate):
    if gate == 'I':
        u = I_mat()
    if gate == 'X':
        u = X_mat()
    elif gate == 'Y':
        u = Y_mat()
    elif gate == 'Z':
        u = Z_mat()
    elif gate == 'H':
        u = H_mat()
    return u


def U_list_dagger(U):
    return U[::-1]


def get_x(vars, ansatz_tree):
    m = len(vars)
    x = 0
    for i in range(m):
        var = vars[i]
        U = ansatz_tree[i]
        # Matrix calculations
        width = len(U[0])
        zeros = zero_state()
        if width > 1:
            for j in range(width - 1):
                zeros = kron(zeros, zero_state())

        U_mat = identity(2 ** width)
        for layer in U:
            U_layer = array([1])
            for j, gate in enumerate(layer):
                u = gate_to_matrix(gate)
                U_layer = kron(U_layer, u)
            U_mat = U_mat @ U_layer
        x += var * U_mat @ zeros
    return x


def get_unitary(U):
    width = len(U[0])

    # Matrix calculations
    U_mat = identity(2 ** width)
    if len(U[0]) == width:
        for layer in U:
            U_layer = array([1])
            for i, gate in enumerate(layer):
                u = gate_to_matrix(gate)
                U_layer = kron(U_layer, u)
            U_mat = U_mat @ U_layer
    return U_mat


def verify_Hadamard_test_result(hardmard_result, U, alpha=1):
    U_mat = get_unitary(U)

    if alpha == 1:
        ideal = real(
            kron(conj(transpose(zero_state())), conj(transpose(zero_state()))) @
            U_mat @ kron(zero_state(), zero_state())).item()
    elif alpha == 1j:
        ideal = imag(
            kron(conj(transpose(zero_state())), conj(transpose(zero_state()))) @
            U_mat @ kron(zero_state(), zero_state())).item()
    else:
        raise ValueError
    # print('The ideal result is:', ideal)

    error = linalg.norm(ideal - hardmard_result)
    print('The error between the Hadamard test result and the ideal result is:', error)


def Hadmard_test_by_matrix(U):
    U_mat = get_unitary(U)
    width = len(U[0])
    zeros = zero_state()
    if width > 1:
        for j in range(width - 1):
            zeros = kron(zeros, zero_state())

    ideal = (conj(transpose(zeros)) @ U_mat @ zeros).item()
    return ideal


def calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=None, shots_budget=1024, frugal=False):
    """
        Please note that the objective function of CVXOPT has the form:   1/2  x^T P x  +   q^T x
        But our objective function is:                                         z^T Q z  - 2 r^T z + 1
        So the coefficients are corrected here.

    :param R:
    :param I:
    :param q:
    :return: Q, r
    """
    A_coeffs = A.get_coeff()
    A_unitaries = A.get_unitary()
    A_terms_number = len(A_coeffs)

    tree_depth = len(ansatz_tree)
    V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')


    if mtd is None:
        mtd = 'Hadamard'

    if mtd == 'Hadamard':
        if frugal is True:
            shots_each_entry = shots_budget / (10 * (tree_depth ** 2 + tree_depth))
            M_A_A = sum(abs(conj(A_coeffs[k]) * A_coeffs[l])
                        for k in range(A_terms_number) for l in range(A_terms_number))
            P_A_A = [10 * int(shots_each_entry * (abs(conj(A_coeffs[k]) * A_coeffs[l]) / M_A_A))
                     for k in range(A_terms_number) for l in range(A_terms_number)]
            M_A = sum([abs(conj(A_coeffs[k]))
                       for k in range(A_terms_number)])
            P_A = [10 * int(shots_each_entry * (abs(conj(A_coeffs[k])) / M_A)) for k in range(A_terms_number)]

        else:
            # shots_ave = int(shots_budget / (((tree_depth ** 2) * (A_terms_number ** 2)) + (tree_depth * A_terms_number)))
            # Uniform distribution
            # P_A_A = [shots_ave for _ in range(A_terms_number) for _ in range(A_terms_number)]
            # P_A = [shots_ave for _ in range(A_terms_number)]
            P_A_A = [12 for _ in range(A_terms_number) for _ in range(A_terms_number)]
            P_A = [12 for _ in range(A_terms_number)]
            # print("Number of shots for Q and r:", shots)
    else:
        P_A_A = None
        P_A = None

    for i in range(tree_depth):
        for j in range(tree_depth):
            # Uniform distribution of the shots
            item = 0
            for k in range(A_terms_number):
                for l in range(A_terms_number):
                    u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
                    if mtd == 'Hadamard':
                        shots = P_A_A[k * A_terms_number + l]
                        inner_product_real = Hadamard_test_ibmq(u, shots=shots)
                        inner_product_imag = Hadamard_test_ibmq(u, alpha=1j, shots=shots)
                        # print(inner_product_real)

                        inner_product = inner_product_real - inner_product_imag * 1j

                    elif mtd == 'Matrix':
                        inner_product = Hadmard_test_by_matrix(u)

                    elif mtd == 'Eigens':
                        inner_product = Hadamard_test_eigens(u)

                    else:
                        raise ValueError

                    item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product

            V_dagger_V[i][j] = item

    R = real(V_dagger_V)
    I = imag(V_dagger_V)

    q = zeros((tree_depth, 1), dtype='complex128')

    for i in range(tree_depth):
        item = 0
        # print_progress((tree_depth * tree_depth + i + 1) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
        for k in range(A_terms_number):
            u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
            if mtd == 'Hadamard':
                shots = P_A[k]
                inner_product_real = Hadamard_test_ibmq(u, shots=shots)
                inner_product_imag = Hadamard_test_ibmq(u, alpha=1j, shots=shots)
                inner_product = inner_product_real - inner_product_imag * 1j

            elif mtd == 'Matrix':
                inner_product = Hadmard_test_by_matrix(u)

            elif mtd == 'Eigens':
                inner_product = Hadamard_test_eigens(u)

            else:
                raise ValueError

            item += conj(A_coeffs[k]) * inner_product

        q[i][0] = item

    # Q     =      R    -I
    #       =      I     R
    Q = array(append(append(R, -I, axis=1), append(I, R, axis=1), axis=0), dtype='float64')

    # r = [Re(q),
    #      Im(q)]
    r_real = real(q)
    r_imag = imag(q)
    r = array(append(r_real, r_imag, axis=0), dtype='float64')
    return Q, r


def calculate_loss_function(A, vars, ansatz_tree, mtd=None, shots_budget=1024, frugal=False):
    A_coeffs = A.get_coeff()
    A_unitaries = A.get_unitary()
    A_terms_number = len(A_coeffs)
    tree_depth = len(ansatz_tree)

    if mtd is None:
        mtd = 'Hadamard'

    if mtd == 'Hadamard':

        if frugal is True:
            shots_budget = shots_budget / 10
            M_Loss = sum([abs(conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l])
                          for i in range(tree_depth)
                          for j in range(tree_depth)
                          for k in range(A_terms_number)
                          for l in range(A_terms_number)] +
                         [2 * abs(real(vars[i]) * A_coeffs[j])
                          for i in range(tree_depth)
                          for j in range(A_terms_number)] +
                         [2 * abs(imag(vars[i]) * A_coeffs[j])
                          for i in range(tree_depth)
                          for j in range(A_terms_number)]
                         )
            P_Loss_term_1 = [10 * int(shots_budget * abs(conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l]) / M_Loss)
                             for i in range(tree_depth)
                             for j in range(tree_depth)
                             for k in range(A_terms_number)
                             for l in range(A_terms_number)]
            P_Loss_term_2 = [10 * int(shots_budget * 2 * abs(real(vars[i]) * A_coeffs[j]) / M_Loss)
                             for i in range(tree_depth)
                             for j in range(A_terms_number)]
            P_Loss_term_3 = [10 * int(shots_budget * 2 * abs(imag(vars[i]) * A_coeffs[j]) / M_Loss)
                             for i in range(tree_depth)
                             for j in range(A_terms_number)]

            #
            # M_Loss = sum([abs(conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l])
            #               for i in range(tree_depth)
            #               for j in range(tree_depth)
            #               for k in range(A_terms_number)
            #               for l in range(A_terms_number)] +
            #              [2 * abs(vars[i]) * A_coeffs[j]
            #               for i in range(tree_depth)
            #               for j in range(A_terms_number)]
            #              )
            #
            # P_Loss_term_1 = [int(shots_budget * abs(conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l]) / M_Loss)
            #                  for i in range(tree_depth)
            #                  for j in range(tree_depth)
            #                  for k in range(A_terms_number)
            #                  for l in range(A_terms_number)]
            # P_Loss_term_2 = [int(shots_budget * 2 * abs(vars[i] * A_coeffs[j]) / M_Loss)
            #                  for i in range(tree_depth)
            #                  for j in range(A_terms_number)]

        else:
            # Uniform distribution
            # shots_ave = int(shots_budget / ((tree_depth ** 2) * (A_terms_number ** 2) + tree_depth * A_terms_number))
            # P_Loss_term_1 = [shots_ave for _ in range((tree_depth ** 2) * (A_terms_number ** 2))]
            # P_Loss_term_2 = [shots_ave for _ in range(tree_depth * A_terms_number)]
            # P_Loss_term_3 = [shots_ave for _ in range(tree_depth * A_terms_number)]
            P_Loss_term_1 = [12 for _ in range((tree_depth ** 2) * (A_terms_number ** 2))]
            P_Loss_term_2 = [12 for _ in range(tree_depth * A_terms_number)]
            P_Loss_term_3 = [12 for _ in range(tree_depth * A_terms_number)]
            # print("Number of shots for loss function:", shots)

        term_1 = 0
        for i in range(tree_depth):
            for j in range(tree_depth):
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        shots = P_Loss_term_1[i * tree_depth * A_terms_number * A_terms_number +
                                         j * A_terms_number * A_terms_number +
                                         k * A_terms_number +
                                         l]
                        u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
                            ansatz_tree[j]
                        inner_product_real = Hadamard_test_ibmq(u, shots=shots)
                        inner_product_imag = Hadamard_test_ibmq(u, alpha=1j, shots=shots)
                        inner_product = inner_product_real - inner_product_imag * 1j
                        term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product

        term_2 = 0
        for i in range(tree_depth):
            for j in range(A_terms_number):
                shots = P_Loss_term_2[i * A_terms_number + j]
                u = A_unitaries[j] + ansatz_tree[i]
                inner_product_real = Hadamard_test_ibmq(u, shots=shots)
                # inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
                # inner_product = inner_product_real - inner_product_imag * 1j
                term_2 += real(vars[i]) * A_coeffs[j] * inner_product_real
                # term_2 += real(vars[i] * A_coeffs[j] * inner_product)

        term_3 = 0
        for i in range(tree_depth):
            for j in range(A_terms_number):
                shots = P_Loss_term_3[i * A_terms_number + j]
                u = A_unitaries[j] + ansatz_tree[i]
                # inner_product_real = Hadamard_test(u, shots=shots)
                inner_product_imag = Hadamard_test_ibmq(u, alpha=1j, shots=shots)
                # inner_product = inner_product_real - inner_product_imag * 1j
                # term_2 += real(vars[i] * A_coeffs[j] * inner_product)
                term_3 += imag(vars[i]) * A_coeffs[j] * inner_product_imag

        loss = term_1 - 2 * (term_2 + term_3) + 1


        # term_2 = 0
        # for i in range(tree_depth):
        #     for j in range(A_terms_number):
        #         shots = P_Loss_term_2[i * A_terms_number + j]
        #         u = A_unitaries[j] + ansatz_tree[i]
        #         inner_product_real = Hadamard_test(u, shots=shots)
        #         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
        #         inner_product = inner_product_real - inner_product_imag * 1j
        #         # term_2 += real(vars[i]) * A_coeffs[j] * inner_product_real
        #         term_2 += real(vars[i] * A_coeffs[j] * inner_product)
        #
        # loss = term_1 - 2 * (term_2) + 1





    elif mtd == 'Matrix':
        A_mat = A.get_matrix()
        x = get_x(vars, ansatz_tree)
        zeros = zero_state()
        width = len(A_unitaries[0][0])
        if width > 1:
            for j in range(width - 1):
                zeros = kron(zeros, zero_state())
        loss = real((conj(transpose(x)) @ conj(transpose(A_mat)) @ A_mat @ x - 2 * real(
            conj(transpose(zeros)) @ A_mat @ x)).item()) + 1

    elif mtd == "Eigens":
        term_1 = 0
        for i in range(tree_depth):
            for j in range(tree_depth):
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
                            ansatz_tree[j]
                        inner_product = Hadamard_test_eigens(u)
                        term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product

        term_2 = 0
        for i in range(tree_depth):
            for j in range(A_terms_number):
                u = A_unitaries[j] + ansatz_tree[i]
                inner_product = Hadamard_test_eigens(u)
                term_2 += real(vars[i] * A_coeffs[j] * inner_product)

        loss = term_1 - 2 * term_2 + 1

    else:
        raise ValueError

    return loss


def verify_loss_function(A, vars, ansatz_tree, loss_es):
    A_mat = A.get_matrix()

    x = get_x(vars, ansatz_tree)

    loss = real((conj(transpose(x)) @ conj(transpose(A_mat)) @ A_mat @ x - 2 * real(
        conj(transpose(zeros)) @ A_mat @ x)).item()) + 1
    print("Loss calculated by matrices:", loss)
    error = linalg.norm(loss_es - loss)
    return error


def verify_gradient_overlap(A, vars, ansatz_tree, max_index_es):
    A_mat = A.get_matrix()
    A_coeffs = A.get_coeff()
    A_unitaries = A.get_unitary()
    A_terms_number = len(A_coeffs)

    x = get_x(vars, ansatz_tree)
    zeros = zero_state()
    width = len(A_unitaries[0][0])
    if width > 1:
        for j in range(width - 1):
            zeros = kron(zeros, zero_state())

    parent_node = ansatz_tree[-1]
    child_space = [parent_node + A_unitaries[i] for i in range(A_terms_number)]
    gradient_overlaps = [0 for _ in range(len(child_space))]

    for i, child_node in enumerate(child_space):
        U_mat = get_unitary(child_node)

        gradient = 2 * conj(transpose(zeros)) @ conj(transpose(U_mat)) @ A_mat @ A_mat @ x - 2 * conj(
            transpose(zeros)) @ conj(transpose(U_mat)) @ A_mat @ zeros
        gradient_overlaps[i] = abs(gradient.item())

    # print('Matrix Calculation:', gradient_overlaps)

    max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]

    if max_index == max_index_es:
        print("Correct!")

    else:
        raise ValueError("Expansion Module is Wrong!")

########################################################################################################################


# def calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=None, shots_power=4):
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
#     A_coeffs = A.get_coeff()
#     A_unitaries = A.get_unitary()
#     A_terms_number = len(A_coeffs)
#
#     tree_depth = len(ansatz_tree)
#     V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')
#
#     if mtd is None:
#         mtd = 'Hadamard'
#
#     for i in range(tree_depth):
#         for j in range(tree_depth):
#             # Unmiti = []
#             # Unmiti_real = []
#             # Unmiti_imag = []
#
#             shots = int(10 ** shots_power)
#             item = 0
#             # print_progress((tree_depth * i + j) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#             for k in range(A_terms_number):
#                 for l in range(A_terms_number):
#                     u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#                     if mtd == 'Hadamard':
#                         inner_product_real = Hadamard_test(u, shots=shots)
#                         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                         inner_product = inner_product_real - inner_product_imag * 1j
#
#                     elif mtd == 'Matrix':
#                         inner_product = Hadmard_test_by_matrix(u)
#
#                     elif mtd == 'Eigens':
#                         inner_product = Hadamard_test(u)
#
#                     item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#
#             V_dagger_V[i][j] = item
#
#             # Shots = []
#             # Unmiti = []
#             # Unmiti_real = []
#             # Unmiti_imag = []
#             #
#             # for p in range(shots_power, 2, -1):
#             #     shots = 10 ** p
#             #     Shots.append(shots)
#             #     unbiased_exp_real = []
#             #     unbiased_exp_imag = []
#             #
#             #     unbiased_exp_number = 10
#             #     for itr in range(unbiased_exp_number):
#             #         #
#             #         #     item = 0
#             #         #     # print_progress((tree_depth * i + j) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#             #         #     for k in range(A_terms_number):
#             #         #         for l in range(A_terms_number):
#             #         #             u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#             #         #             if mtd == 'Hadamard':
#             #         #                 inner_product_real = Hadamard_test(u, shots=shots)
#             #         #                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#             #         #                 inner_product = inner_product_real - inner_product_imag * 1j
#             #         #
#             #         #             elif mtd == 'Matrix':
#             #         #                 inner_product = Hadmard_test_by_matrix(u)
#             #         #
#             #         #             elif mtd == 'Eigens':
#             #         #                 inner_product = Hadamard_test(u)
#             #         #
#             #         #             item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#             #         #
#             #         #     unbiased_exp.append(item)
#             #         #
#             #         # exp_obs_ave = sum(unbiased_exp) / unbiased_exp_number
#             #         # unbiased_exp_half = [i for i in unbiased_exp if i >= exp_obs_ave]
#             #         # unbiased_exp_half_exp = sum(unbiased_exp_half) / len(unbiased_exp_half)
#             #         # # if exp_obs_ave > real_exp_std:
#             #         # Shots.append(shot)
#             #         # Exp.append(unbiased_exp_half_exp)
#             #
#             #         item = 0
#             #         # print_progress((tree_depth * i + j) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#             #         for k in range(A_terms_number):
#             #             for l in range(A_terms_number):
#             #                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
#             #                     ansatz_tree[j]
#             #                 if mtd == 'Hadamard':
#             #                     inner_product_real = Hadamard_test(u, shots=shots)
#             #                     inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#             #                     inner_product = inner_product_real - inner_product_imag * 1j
#             #
#             #                 elif mtd == 'Matrix':
#             #                     inner_product = Hadmard_test_by_matrix(u)
#             #
#             #                 elif mtd == 'Eigens':
#             #                     inner_product = Hadamard_test(u)
#             #
#             #                 item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#             #
#             #         unbiased_exp_real.append(real(item))
#             #         unbiased_exp_imag.append(imag(item))
#             #
#             #     exp_obs_real_ave = sum(unbiased_exp_real) / unbiased_exp_number
#             #     unbiased_exp_real_half = [i for i in unbiased_exp_real if i >= exp_obs_real_ave]
#             #     unbiased_exp_real_half_exp = sum(unbiased_exp_real_half) / len(unbiased_exp_real_half)
#             #     Unmiti_real.append(unbiased_exp_real_half_exp)
#             #
#             #     exp_obs_imag_ave = sum(unbiased_exp_imag) / unbiased_exp_number
#             #     unbiased_exp_imag_half = [i for i in unbiased_exp_imag if i >= exp_obs_imag_ave]
#             #     unbiased_exp_imag_half_exp = sum(unbiased_exp_imag_half) / len(unbiased_exp_imag_half)
#             #     Unmiti_imag.append(unbiased_exp_imag_half_exp)
#             #
#             # miti_exp_real = Richardson_extrapolate(Shots, Unmiti_real)
#             # miti_exp_imag = Richardson_extrapolate(Shots, Unmiti_imag)
#             #
#             # miti_exp = miti_exp_real + miti_exp_imag * 1j
#             # V_dagger_V[i][j] = miti_exp
#
#
#
#
#
#
#
#
#
#     R = real(V_dagger_V)
#     I = imag(V_dagger_V)
#
#     q = zeros((tree_depth, 1), dtype='complex128')
#
#     for i in range(tree_depth):
#         shots = int(10 ** shots_power)
#
#         item = 0
#         # print_progress((tree_depth * tree_depth + i + 1) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#         for k in range(A_terms_number):
#             u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#             if mtd == 'Hadamard':
#                 inner_product_real = Hadamard_test(u, shots=shots)
#                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                 inner_product = inner_product_real - inner_product_imag * 1j
#
#             elif mtd == 'Matrix':
#                 inner_product = Hadmard_test_by_matrix(u)
#
#             elif mtd == 'Eigens':
#                 inner_product = Hadamard_test(u)
#
#             item += conj(A_coeffs[k]) * inner_product
#
#         q[i][0] = item
#
#
#
#
#
#
#     # for i in range(tree_depth):
#     #     Shots = []
#     #     Unmiti = []
#     #     Unmiti_real = []
#     #     Unmiti_imag = []
#     #
#     #     for p in range(shots_power, 2, -1):
#     #         shots = 10 ** p
#     #         Shots.append(shots)
#     #
#     #         unbiased_exp_real = []
#     #         unbiased_exp_imag = []
#     #
#     #         unbiased_exp_number = 10
#     #         for itr in range(unbiased_exp_number):
#     #
#     #             item = 0
#     #             # print_progress((tree_depth * tree_depth + i + 1) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#     #             for k in range(A_terms_number):
#     #                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#     #                 if mtd == 'Hadamard':
#     #                     inner_product_real = Hadamard_test(u, shots=shots)
#     #                     inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#     #                     inner_product = inner_product_real - inner_product_imag * 1j
#     #
#     #                 elif mtd == 'Matrix':
#     #                     inner_product = Hadmard_test_by_matrix(u)
#     #
#     #                 elif mtd == 'Eigens':
#     #                     inner_product = Hadamard_test(u)
#     #
#     #                 item += conj(A_coeffs[k]) * inner_product
#     #
#     #             unbiased_exp_real.append(real(item))
#     #             unbiased_exp_imag.append(imag(item))
#     #
#     #         exp_obs_real_ave = sum(unbiased_exp_real) / unbiased_exp_number
#     #         unbiased_exp_real_half = [i for i in unbiased_exp_real if i >= exp_obs_real_ave]
#     #         unbiased_exp_real_half_exp = sum(unbiased_exp_real_half) / len(unbiased_exp_real_half)
#     #         Unmiti_real.append(unbiased_exp_real_half_exp)
#     #
#     #         exp_obs_imag_ave = sum(unbiased_exp_imag) / unbiased_exp_number
#     #         unbiased_exp_imag_half = [i for i in unbiased_exp_imag if i >= exp_obs_imag_ave]
#     #         unbiased_exp_imag_half_exp = sum(unbiased_exp_imag_half) / len(unbiased_exp_imag_half)
#     #         Unmiti_imag.append(unbiased_exp_imag_half_exp)
#     #
#     #
#     #     miti_exp_real = Richardson_extrapolate(Shots, Unmiti_real)
#     #     miti_exp_imag = Richardson_extrapolate(Shots, Unmiti_imag)
#     #
#     #     miti_exp = miti_exp_real + miti_exp_imag * 1j
#     #
#     #
#     #     q[i][0] = miti_exp
#
#
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
#     return Q, r


# def calculate_loss_function(A, vars, ansatz_tree, mtd=None, shots_power=4):
#     A_coeffs = A.get_coeff()
#     A_unitaries = A.get_unitary()
#     A_terms_number = len(A_coeffs)
#     tree_depth = len(ansatz_tree)
#
#     if mtd is None:
#         mtd = 'Hadamard'
#
#     if mtd == 'Hadamard':
#
#         shots = int(10 ** shots_power)
#
#
#         term_1 = 0
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#                         inner_product_real = Hadamard_test(u, shots=shots)
#                         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                         inner_product = inner_product_real - inner_product_imag * 1j
#                         term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#
#         term_2 = 0
#         for i in range(tree_depth):
#             for j in range(A_terms_number):
#                 u = A_unitaries[j] + ansatz_tree[i]
#                 inner_product_real = Hadamard_test(u, shots=shots)
#                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                 inner_product = inner_product_real - inner_product_imag * 1j
#                 term_2 += real(vars[i] * A_coeffs[j] * inner_product)
#         loss = term_1 - 2 * term_2 + 1
#
#         # if mtd == 'Hadamard':
#         #     Shots = []
#         #     Unmiti = []
#         #     Unmiti_real = []
#         #     Unmiti_imag = []
#         #
#         #     for p in range(shots_power, 2, -1):
#         #         shots = 10 ** p
#         #         Shots.append(shots)
#         #
#         #         unbiased_exp_real = []
#         #         unbiased_exp_imag = []
#         #
#         #         unbiased_exp_number = 10
#         #         for itr in range(unbiased_exp_number):
#         #
#         #             term_1 = 0
#         #             for i in range(tree_depth):
#         #                 for j in range(tree_depth):
#         #                     for k in range(A_terms_number):
#         #                         for l in range(A_terms_number):
#         #                             u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
#         #                                 ansatz_tree[j]
#         #                             inner_product_real = Hadamard_test(u, shots=shots)
#         #                             inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#         #                             inner_product = inner_product_real - inner_product_imag * 1j
#         #                             term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#         #
#         #             term_2 = 0
#         #             for i in range(tree_depth):
#         #                 for j in range(A_terms_number):
#         #                     u = A_unitaries[j] + ansatz_tree[i]
#         #                     inner_product_real = Hadamard_test(u, shots=shots)
#         #                     inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#         #                     inner_product = inner_product_real - inner_product_imag * 1j
#         #                     term_2 += real(vars[i] * A_coeffs[j] * inner_product)
#         #             loss = term_1 - 2 * term_2 + 1
#         #
#         #             unbiased_exp_real.append(real(loss))
#         #             unbiased_exp_imag.append(imag(loss))
#         #
#         #         exp_obs_real_ave = sum(unbiased_exp_real) / unbiased_exp_number
#         #         unbiased_exp_real_half = [i for i in unbiased_exp_real if i >= exp_obs_real_ave]
#         #         unbiased_exp_real_half_exp = sum(unbiased_exp_real_half) / len(unbiased_exp_real_half)
#         #         Unmiti_real.append(unbiased_exp_real_half_exp)
#         #
#         #         exp_obs_imag_ave = sum(unbiased_exp_imag) / unbiased_exp_number
#         #         unbiased_exp_imag_half = [i for i in unbiased_exp_imag if i >= exp_obs_imag_ave]
#         #         unbiased_exp_imag_half_exp = sum(unbiased_exp_imag_half) / len(unbiased_exp_imag_half)
#         #         Unmiti_imag.append(unbiased_exp_imag_half_exp)
#         #
#         #     miti_exp_real = Richardson_extrapolate(Shots, Unmiti_real)
#         #     miti_exp_imag = Richardson_extrapolate(Shots, Unmiti_imag)
#         #
#         #     loss = miti_exp_real + miti_exp_imag * 1j
#
#
#
#     elif mtd == 'Matrix':
#         A_mat = A.get_matrix()
#         x = get_x(vars, ansatz_tree)
#         zeros = zero_state()
#         width = len(A_unitaries[0][0])
#         if width > 1:
#             for j in range(width - 1):
#                 zeros = kron(zeros, zero_state())
#         loss = real((conj(transpose(x)) @ conj(transpose(A_mat)) @ A_mat @ x - 2 * real(conj(transpose(zeros)) @ A_mat @ x)).item()) + 1
#
#     elif mtd == "Eigens":
#         term_1 = 0
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#                         inner_product = Hadamard_test(u)
#                         term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#
#         term_2 = 0
#         for i in range(tree_depth):
#             for j in range(A_terms_number):
#                 u = A_unitaries[j] + ansatz_tree[i]
#                 inner_product = Hadamard_test(u)
#                 term_2 += real(vars[i] * A_coeffs[j] * inner_product)
#         loss = term_1 - 2 * term_2 + 1
#
#     return loss
