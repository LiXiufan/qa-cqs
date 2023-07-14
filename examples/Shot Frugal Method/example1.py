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
    Test the shot frugal method for the summation of different expectations obtained by Hadamard tests.
    Set O = \sum_m^{M} c_m * O_m
    Goal: Estimate the O with a limited number of shots.
"""

from generator import CoeffMatrix
from numpy import array, linalg
from numpy import log2, log10
from numpy import sqrt, kron, transpose, conj, real, imag, append
from cqs_module.optimization import solve_combination_parameters
from cqs_module.calculation import calculate_Q_r_by_Hadamrd_test, gate_to_matrix, zero_state, calculate_loss_function, \
    verify_loss_function
from cqs_module.expansion import expand_ansatz_tree
from utils import write_running_data

import matplotlib.pyplot as plt
from hardware.Qibo.qibo_access import Hadamard_test





























########################################################################################################################
#                                           CODES OF OLD VERSIONS
########################################################################################################################
#
#
# # Assume states are [U1, U2, U3], calculate R and I
# m = 3
# U1 = [unitaries[0]]
# # From right to the left operating to zero state
# U2 = [unitaries[0], unitaries[1]]
# U3 = [unitaries[2]]
# # print(unitaries)
# states = [U1, U2, U3]
# # print(states)
#
#
# V_dagger_V = zeros((m, m), dtype='complex128')
#
# for i in range(m):
#     for j in range(m):
#         item = 0
#         print_progress((m * i + j) / ((m * m) + m), 'Current Progress:')
#         for k in range(number_of_terms):
#             for l in range(number_of_terms):
#                 u = U_list_dagger(states[i]) + U_list_dagger([unitaries[k]]) + [unitaries[l]] + states[j]
#                 inner_product_real = Hadamard_test(u, shots=shots)
#                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                 inner_product = inner_product_real - inner_product_imag * 1j
#                 item += conj(coeffs[k]) * coeffs[l] * inner_product
#         V_dagger_V[i][j] = item
# R = real(V_dagger_V)
# I = imag(V_dagger_V)
#
# q = zeros((m, 1), dtype='complex128')
# for i in range(m):
#     item = 0
#     print_progress((m * m + i + 1) / ((m * m) + m), 'Current Progress:')
#     for k in range(number_of_terms):
#         u = U_list_dagger(states[i]) + U_list_dagger([unitaries[k]])
#         inner_product_real = Hadamard_test(u, shots=shots)
#         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#         inner_product = inner_product_real + inner_product_imag * 1j
#         item += conj(coeffs[k]) * inner_product
#     q[i][0] = item
#
# # print(q)
#
#
# Q, r = calculate_Q_and_r(R, I, q)


# print('Hadamard test result:', R)
# print('Hadamard test result:', I)
# print('Hadamard test result:', q)


# V = append(mat @ kron(gate_to_matrix(unitaries[0][0]), gate_to_matrix(unitaries[0][1])) @ kron(zero_state(), zero_state()),
#            mat @ kron(gate_to_matrix(unitaries[0][0]), gate_to_matrix(unitaries[0][1])) @ kron(gate_to_matrix(unitaries[1][0]), gate_to_matrix(unitaries[1][1])) @ kron(zero_state(), zero_state()), axis=1)
# V = append(V, mat @ kron(gate_to_matrix(unitaries[2][0]), gate_to_matrix(unitaries[2][1])) @ kron(zero_state(), zero_state()), axis=1)
# V = append(V, mat @ mat @ mat @ mat @ b.reshape(dim, 1), axis=1)

# R_idea = real(conj(transpose(V)) @ V)
# I_idea = imag(conj(transpose(V)) @ V)
# q_idea = (conj(transpose(V)) @ b).reshape(m, 1)
# print("Ideal result:", R_idea)
# print("Ideal result:", I_idea)
# print('Ideal result:', q_idea)


#
# print('Error between the Hadamard one and the ideal one:', linalg.norm(R - R_idea))
# print('Error between the Hadamard one and the ideal one:', linalg.norm(I - I_idea))
# print(q - q_idea)
# print('Error between the Hadamard one and the ideal one:', linalg.norm(q - q_idea))


# q = (conj(transpose(V)) @ b.reshape(dim, 1)).reshape(4)

# Calculate Q and r using the R, I, and q obatined from the sampling outcomes.
# Q, r = calculate_Q_and_r(R, I, q)


# vars = solve_combination_parameters(Q, r)

# x = vars[0] * b.reshape(dim, 1) + vars[1] * mat @ b.reshape(dim, 1) + \
#     vars[2] * mat @ mat @ b.reshape(dim, 1) + vars[3] * mat @ mat @ mat @ b.reshape(dim, 1)
#
# error = linalg.norm(analy_x-x)
# print('The error between analytical solution and the CQS solution is:', error)
