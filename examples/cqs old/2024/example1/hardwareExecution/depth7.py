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

from cqs.object import CoeffMatrix
from numpy import real, array
from cqs.optimization import solve_combination_parameters
from cqs.local.calculation import calculate_Q_r_by_Hadamrd_test
from cqs.verifier import get_unitary

qubit_number = 5
dim = 2 ** qubit_number
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 4
ITR = 7

# Total Budget of shots
shots_total_budget = 10 ** 6
# Expected error
error = 0.1

# Initialize the coefficient matrix
# print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])
# Generate A with the following way (problem 8)
coeffs = [0.358,
          0.011,
          -0.919,
          -0.987]
unitaries = [[['I', 'Y', 'X', 'I', 'Y']],
             [['Z', 'X', 'Y', 'X', 'I']],
             [['X', 'X', 'Z', 'I', 'X']],
             [['Z', 'I', 'X', 'Z', 'Z']]]
u_b = [['I', 'I', 'I', 'I', 'I']]

file_name = 'example1depth7.txt'

# Initialize the coefficient matrix
A = CoeffMatrix(number_of_terms, dim, qubit_number)
A.generate(which_form='Unitaries', given_unitaries=unitaries, given_coeffs=coeffs)
B = sum([abs(coeff) for coeff in coeffs])
# Values on the right hand side of the equation.
u_b_mat = get_unitary(u_b)
zeros = array([1] + [0 for _ in range(dim - 1)])
b_mat = u_b_mat @ zeros

file1 = open(file_name, "a")
file1.writelines(['Coefficients of the terms are:', str(coeffs), '\n'])
file1.writelines(['Decomposed unitaries are:', str(unitaries), '\n'])
file1.writelines(['The vector on the right hand side of the equation is:', str(b_mat), '\n'])
file1.close()

# 2. Initialization
# Total iteration
batch = shots_total_budget / (sum(list(i ** 2 for i in range(1, ITR + 1))))
# Use a wise shot allocation method to display the shots
Q_r_budgets = []
loss_budgets = []
gradient_budgets = []
for itr in range(1, ITR + 1):
    weight = itr ** 2
    shot_budget = weight * batch

    # Shot budget to calculate the matrix Q and vector r
    Q_r_budget = int(0.4 * shot_budget)
    Q_r_budgets.append(Q_r_budget)

    # Shot budget to calculate the loss function
    loss_budget = int(0.4 * shot_budget)
    loss_budgets.append(loss_budget)

    # Shot budget to calculate the gradient overlaps
    gradient_budget = int(0.2 * shot_budget)
    gradient_budgets.append(gradient_budget)


# Customize the Ansatz tree and solve the probelm given a certain type
itr = 7
ansatz_tree = [[['I', 'I', 'I', 'I', 'I']],
               [['I', 'I', 'I', 'I', 'I'], ['Z', 'I', 'X', 'Z', 'Z']],
               [['I', 'I', 'I', 'I', 'I'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y']],
               [['I', 'I', 'I', 'I', 'I'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y'], ['Z', 'I', 'X', 'Z', 'Z']],
               [['I', 'I', 'I', 'I', 'I'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y']],
               [['I', 'I', 'I', 'I', 'I'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y'], ['X', 'X', 'Z', 'I', 'X']],
               [['I', 'I', 'I', 'I', 'I'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y'], ['Z', 'I', 'X', 'Z', 'Z'], ['I', 'Y', 'X', 'I', 'Y'], ['X', 'X', 'Z', 'I', 'X'], ['Z', 'X', 'Y', 'X', 'I']]]

backend = 'braket'
TASKS = 0
SHOTS = 0
frugal = None
file1 = open(file_name, "a")
file1.writelines(['\n', "Itr:", str(itr), " Ansatz tree is:", str(ansatz_tree), '\n\n'])
file1.close()

Q, r, TASKS, SHOTS = calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, backend=backend,
                                                           shots_budget=Q_r_budgets[itr - 1], frugal=frugal,
                                                           tasks_num=TASKS, shots_num=SHOTS,
                                                           file_name=file_name)
file1 = open(file_name, "a")
file1.writelines(
    ['\n', "Itr:", str(itr), " Matrix Q is:\n", str(Q), "\nItr:", str(itr), " Vector r is:\n",
     str(r), '\n\n'])
file1.close()
loss, vars = solve_combination_parameters(Q, r, which_opt=None)
file1 = open(file_name, "a")
file1.writelines(["\nItr:", str(itr), " Combination parameters are:", str(vars), '\n\n'])
file1.writelines(['\nItr:', str(itr), " Loss:", str(loss), '\n\n'])
file1.close()






