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
from qiskit_ibm_provider import IBMProvider

import matplotlib.pyplot as plt

# from hardware.IonQ.qiskit_access import Hadamard_test

# Generate the problem of solving a linear systems of equations.
# Set the dimension of the coefficient matrix A on the left hand side of the equation.
qubit_number = 5

dim = 2 ** qubit_number
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 4
# Total Budget of shots
# shots_total_budget = 500

# shots_total_budget = 5 * 10 ** 4
shots_total_budget = 6996
# shots_total_budget = 10 ** 5

# total_tree_depth = 50
error = 0.1

# mtd = 'Matrix'
mtd = 'Hadamard'
# mtd = 'Eigens'

# Initialize the coefficient matrix
A = CoeffMatrix(number_of_terms, dim, qubit_number)
print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

# Generate A with Pauli matrices
# A.generate()

# Get the coefficients of the terms and the unitaries
# coeffs = A.get_coeff()
# unitaries = A.get_unitary()


# coeffs = [1, 0.2, 0.2]
# coeffs = [1, 0.73, 0.2]
# coeffs = [1, 0.1, -0.5, 0.1]
# unitaries = [[['I', 'I', 'I', 'I', 'I']], [['I', 'X', 'I', 'X', 'I']], [['X', 'Z', 'I', 'I', 'Z']], [['Y', 'I', 'Z', 'Y', 'I']]]
# unitaries = [[['I', 'I', 'I', 'I', 'I']], [['X', 'Z', 'I', 'I', 'I']], [['X', 'I', 'Z', 'I', 'I']]]

coeffs = [1, 0.5, 0.2]
unitaries = [[['I', 'I', 'I', 'I', 'I']], [['X', 'Z', 'I', 'I', 'I']], [['X', 'I', 'Z', 'I', 'I']]]
# Number of Hadamard tests in total: 636

# coeffs = [0.5, -0.11037169750027553, 0.8186265152730448, -0.868868734050515]
# coeffs =  [0.3582017357918057, 0.011037169750027553, -0.9186265152730448, -0.9868868734050515]
# unitaries = [[['I', 'Y', 'X', 'I', 'Y']], [['Z', 'X', 'Y', 'X', 'I']], [['X', 'X', 'Z', 'I', 'X']], [['Z', 'I', 'X', 'Z', 'Z']]]
A.generate(which_form='Unitaries', given_unitaries=unitaries, given_coeffs=coeffs)


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
ITR = 3
# batch = int(shots_total_budget / (sum(list((2 * i) * number_of_terms * number_of_terms for i in range(1, ITR + 1)))))
batch = shots_total_budget / (sum(list(i ** 2 for i in range(1, ITR + 1))))

# xiufan_token = 'd383594cc2f9b3c7baf90b70b11f658c2df3ede2f534e571cc73d5e01f70e7a23620b4384594db7660ded289e864f137f3be56f05f39b9a113545ad90b11a302'
# IBMProvider.save_account(xiufan_token)

def Shot_Frugal_Main():

    # At the begining, the ansatz tree only contains |b>, so we define it as [[['I', 'I']]]
    ansatz_tree = [[['I' for _ in range(qubit_number)]]]

    Itr = []
    loss_list_hadamard_frugal = []
    itr = 0
    loss = 1

    for _ in range(ITR):

        itr += 1
        weight = itr ** 2
        shot_budget = weight * batch

        Q_r_budget = int(0.4 * shot_budget)
        loss_budget = int(0.4 * shot_budget)
        gradient_budget = int(shot_budget - loss_budget - Q_r_budget)
        print(Q_r_budget)
        print(loss_budget)
        print(gradient_budget)

        Itr.append(itr)
        eg = "Regression loss of depth" + str(itr)
        # Performing Hadamard test to calculate Q and r
        mtd = 'Hadamard'
        # mtd = 'Eigens'
        Q, r = calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=mtd, shots_budget=Q_r_budget, frugal=True)

        # Estimate the property of Q and r
        # print('B:', B)
        # print((itr ** 3) * (B ** 4) * (number_of_terms ** 2))
        # measurement_shots = (itr ** 3) * (B ** 4) * (number_of_terms ** 2) *\
        #                     linalg.norm(Q) * (linalg.norm(linalg.inv(Q))) ** 2 * \
        #                     (1 + linalg.norm(linalg.inv(Q / 2) @ (r / (-2))) ** 2) / error
        # print('Proposition 2 guarantees that the measurement shots should be more than:', int(measurement_shots.item()) + 1)

        # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
        vars = solve_combination_parameters(Q, r, which_opt=None)
        # vars = solve_combination_parameters(Q, r, which_opt=None)
        # Calculate the regression loss function to test if it is in the error range
        # loss = calculate_loss_function(A, vars, ansatz_tree, shots=shots)
        # print('Tree Depth:', itr, "test:", test)

        # loss = real(calculate_loss_function(A, vars, ansatz_tree, shots=shots))
        mtd = 'Hadamard'
        # mtd = 'Eigens'
        loss = abs(real(calculate_loss_function(A, vars, ansatz_tree, mtd=mtd, shots_budget=loss_budget, frugal=True)))
        print('Tree Depth:', itr, "Loss:", loss)
        loss_list_hadamard_frugal.append(loss)

        # print("Error between loss_estimated and loss_ideal is:", loss_error)
        # print()

        # if itr % 10 == 0:
        #     print("tree depth:", itr, "  loss:", "%.4f" % loss)

        # If the loss is in our error range, stop the iteration;
        # if abs(loss) <= error:
        #     print(f"Find the optimal solution of combination parameters \n {vars} \n and the Ansatz tree structure \n {ansatz_tree}")
        #     break
        #
        # else:
        #     anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_power=shots_power)
        mtd = 'Hadamard'
        # mtd = 'Eigens'
        anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_budget=gradient_budget, frugal=True)
    return Itr, loss_list_hadamard_frugal

def Shot_Not_Frugal_Main():

    ansatz_tree = [[['I' for _ in range(qubit_number)]]]

    Itr = []
    loss_list_hadamard = []
    itr = 0
    loss = 1
    for _ in range(ITR):

        itr += 1
        weight = itr ** 2
        shot_budget = weight * batch
        Q_r_budget = int(0.4 * shot_budget)
        loss_budget = int(0.4 * shot_budget)
        gradient_budget = int(shot_budget - loss_budget - Q_r_budget)


        Itr.append(itr)
        eg = "Regression loss of depth" + str(itr)
        # Performing Hadamard test to calculate Q and r
        mtd = 'Hadamard'
        Q, r = calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=mtd, shots_budget=Q_r_budget, frugal=False)

        # Estimate the property of Q and r
        # print('B:', B)
        # print((itr ** 3) * (B ** 4) * (number_of_terms ** 2))
        # measurement_shots = (itr ** 3) * (B ** 4) * (number_of_terms ** 2) *\
        #                     linalg.norm(Q) * (linalg.norm(linalg.inv(Q))) ** 2 * \
        #                     (1 + linalg.norm(linalg.inv(Q / 2) @ (r / (-2))) ** 2) / error
        # print('Proposition 2 guarantees that the measurement shots should be more than:', int(measurement_shots.item()) + 1)

        # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
        vars = solve_combination_parameters(Q, r, which_opt=None)

        # vars = solve_combination_parameters(Q, r, which_opt=None)
        # Calculate the regression loss function to test if it is in the error range
        # loss = calculate_loss_function(A, vars, ansatz_tree, shots=shots)
        # print('Tree Depth:', itr, "test:", test)

        # loss = real(calculate_loss_function(A, vars, ansatz_tree, shots=shots))
        mtd = 'Hadamard'
        loss = abs(real(calculate_loss_function(A, vars, ansatz_tree, mtd=mtd, shots_budget=loss_budget, frugal=False)))
        print('Tree Depth:', itr, "Loss:", loss)
        loss_list_hadamard.append(loss)

        # print("Error between loss_estimated and loss_ideal is:", loss_error)
        # print()

        # if itr % 10 == 0:
        #     print("tree depth:", itr, "  loss:", "%.4f" % loss)

        # If the loss is in our error range, stop the iteration;
        # if abs(loss) <= error:
        #     print(f"Find the optimal solution of combination parameters \n {vars} \n and the Ansatz tree structure \n {ansatz_tree}")
        #     break
        #
        # else:
        #     anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_power=shots_power)
        mtd = 'Hadamard'
        anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_budget=gradient_budget, frugal=False)

    return Itr, loss_list_hadamard

def Matrix_Multiplication_Main():


    ansatz_tree = [[['I' for _ in range(qubit_number)]]]


    Itr = []
    loss_list_matrix = []
    itr = 0
    loss = 1
    mtd = 'Eigens'
    # if qubit_number <= 10:
    #     A_mat = A.get_matrix()
    #     con_num = linalg.norm(A_mat) * linalg.norm(linalg.inv(A_mat))
    #     rig_guar_mea = 10 ** 5 * (B ** 4) * (number_of_terms ** 2) * (con_num ** 2) / ((error) ** 5)
    #     print(rig_guar_mea)
    # Record the loss information

    # for itr in range(1, total_tree_depth + 1):
    # while abs(loss) >= error:
    for _ in range(ITR):

        itr += 1
        shot_budget = itr * batch
        Itr.append(itr)
        eg = "Regression loss of depth" + str(itr)

        Q, r = calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=mtd, shots_budget=shot_budget, frugal=False)

        # Estimate the property of Q and r
        # print('B:', B)
        # print((itr ** 3) * (B ** 4) * (number_of_terms ** 2))
        # measurement_shots = (itr ** 3) * (B ** 4) * (number_of_terms ** 2) *\
        #                     linalg.norm(Q) * (linalg.norm(linalg.inv(Q))) ** 2 * \
        #                     (1 + linalg.norm(linalg.inv(Q / 2) @ (r / (-2))) ** 2) / error
        # print('Proposition 2 guarantees that the measurement shots should be more than:', int(measurement_shots.item()) + 1)

        # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
        vars = solve_combination_parameters(Q, r, which_opt=None)
        # vars = solve_combination_parameters(Q, r, which_opt=None)
        # Calculate the regression loss function to test if it is in the error range
        # loss = calculate_loss_function(A, vars, ansatz_tree, shots=shots)
        # print('Tree Depth:', itr, "test:", test)

        # loss = real(calculate_loss_function(A, vars, ansatz_tree, shots=shots))
        # mtd = 'Eigens'
        loss = real(calculate_loss_function(A, vars, ansatz_tree, mtd=mtd, shots_budget=shot_budget, frugal=False))
        print('Tree Depth:', itr, "Loss:", loss)
        loss_list_matrix.append(loss)

        # print("Error between loss_estimated and loss_ideal is:", loss_error)
        # print()

        # if itr % 10 == 0:
        #     print("tree depth:", itr, "  loss:", "%.4f" % loss)

        # If the loss is in our error range, stop the iteration;
        # if abs(loss) <= error:
        #     print(f"Find the optimal solution of combination parameters \n {vars} \n and the Ansatz tree structure \n {ansatz_tree}")
        #     break
        #
        # else:
        #     anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_power=shots_power)

        anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_budget=shot_budget, frugal=False)

    return Itr, loss_list_matrix

# Itr, loss_list_hadamard_frugal = Shot_Frugal_Main()
Itr, loss_list_hadamard = Shot_Not_Frugal_Main()
_, loss_list_matrix = Matrix_Multiplication_Main()

plt.title("CQS: Loss - Depth")

plt.plot(Itr, [0 for _ in Itr], 'b--')
# plt.plot(Itr, loss_list_hadamard_frugal, 'g-', linewidth=2.5, label='Loss Function by Hadamard Tests with frugal method')
plt.plot(Itr, loss_list_hadamard, '-', color='blue', linewidth=2.5, label='Loss Function by Hadamard Tests')
plt.plot(Itr, loss_list_matrix, '-', color='red', linewidth=2.5, label='Loss Function by Matrix Multiplication')

plt.legend()
plt.xlabel("Depth")
plt.ylabel("Loss")
plt.show()





















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
