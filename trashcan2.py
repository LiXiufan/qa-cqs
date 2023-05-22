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
    Execution of the program.
"""

from generator import CoeffMatrix
from numpy import array, linalg
from numpy import log2, log10
from numpy import sqrt, kron, transpose, conj, real, imag, append
from cqs_module.optimization import solve_combination_parameters
from cqs_module.calculator import calculate_Q_r_by_Hadamrd_test, gate_to_matrix, zero_state, calculate_loss_function, verify_loss_function
from cqs_module.expansion import expand_ansatz_tree
from utils import write_running_data

import matplotlib.pyplot as plt

# from hardware.IonQ.qiskit_access import Hadamard_test

# Generate the problem of solving a linear systems of equations.
# Set the dimension of the coefficient matrix A on the left hand side of the equation.
qubit_number = 4

dim = 2 ** qubit_number
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 4
shots_power = 4

# total_tree_depth = 50
error = 1

# mtd = 'Matrix'
mtd = 'Hadamard'
# mtd = 'Eigens'

# Initialize the coefficient matrix
A = CoeffMatrix(number_of_terms, dim, qubit_number)
print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

# Generate A with Pauli matrices
A.generate()

# Generate A with other forms (Haar matrices)
# A.generate('Haar')

# Get the coefficients of the terms and the unitaries
coeffs = A.get_coeff()
unitaries = A.get_unitary()

# coeffs =  [1.423674376436618, 0.5788003037890421, 1.7368252690727974, 1.6660186975242612, 1.1814000196931385]
# unitaries =  [[['X', 'Z', 'X', 'Y', 'I', 'X', 'X']], [['Y', 'Z', 'I', 'Z', 'Z', 'X', 'Z']], [['I', 'X', 'I', 'Z', 'X', 'X', 'Y']], [['Y', 'X', 'I', 'Y', 'X', 'I', 'Y']], [['Z', 'I', 'X', 'X', 'X', 'Y', 'Z']]]
# A.generate(which_form='Unitaries', given_unitaries=unitaries, given_coeffs=coeffs)

print('Coefficients of the terms are:', coeffs)
print('Decomposed unitaries are:', unitaries)
B = sum([abs(coeff) for coeff in coeffs])

# Values on the right hand side of the equation.
# b = array([1] + [0 for _ in range(dim - 1)])
# print('The vector on the right hand side of the equation is:', b)

# Analytical solution, we can get the matrix form of A
# mat = A.get_matrix()
# print('The coefficient matrix is:/n', mat)
# analy_x = linalg.inv(mat) @ b.reshape(dim, 1)
# print('The analytical solution of the linear systems of equations is:', analy_x)

# At the begining, the ansatz tree only contains |b>, so we define it as [[['I', 'I']]]
ansatz_tree = [[['I' for _ in range(qubit_number)]]]

Itr = []
loss_list = []
itr = 0
loss = 1

if qubit_number <= 10:
    A_mat = A.get_matrix()
    con_num = linalg.norm(A_mat) * linalg.norm(linalg.inv(A_mat))
    # print("B:", B)
    print("condition number is:", con_num)
    rig_guar_mea = (4 ** 5) * (B ** 4) * (number_of_terms ** 2) * (con_num ** 2) / ((error) ** 5)
    print(rig_guar_mea)
# Record the loss information
