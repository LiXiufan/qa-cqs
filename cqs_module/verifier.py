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
    This is the verifier module to verify the results of Hadamard tests and gradient overlaps
"""

from numpy import array, sqrt, identity
from numpy import kron, conj, transpose
from numpy import real, imag
from numpy import linalg

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



def verify_loss_function(A, vars, ansatz_tree, loss_es):
    A_mat = A.get_matrix()
    A_unitaries = A.get_unitary()

    x = get_x(vars, ansatz_tree)
    zeros = zero_state()
    width = len(A_unitaries[0][0])
    if width > 1:
        for j in range(width - 1):
            zeros = kron(zeros, zero_state())

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
