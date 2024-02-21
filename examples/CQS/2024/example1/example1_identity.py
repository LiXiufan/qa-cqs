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
    An example of customized linear systems of equations.

"""

from examples.CQS.cqs_main import EXE

qubit_number = 5
dim = 2 ** qubit_number
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 4
ITR = 10

# Total Budget of shots
shots_total_budget = 10 ** 6
# Expected error
error = 0.1

# Initialize the coefficient matrix
# print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

# Generate A with the following way
coeffs = [0.358,
          0.011,
          -0.919,
          -0.987]
unitaries = [[['I', 'Y', 'X', 'I', 'Y']],
             [['Z', 'X', 'Y', 'X', 'I']],
             [['X', 'X', 'Z', 'I', 'X']],
             [['Z', 'I', 'X', 'Z', 'Z']]]
u_b = [['I', 'I', 'I', 'I', 'I']]

file_name = 'example1test.txt'
EXE(qubit_number, number_of_terms, ITR, coeffs, unitaries, u_b, file_name, backend='matrix', expan_mtd='gradient')







