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

from cqs.optimization import solve_combination_parameters
from cqs.local.calculation import calculate_Q_r
from cqs.expansion import expand_ansatz_tree
from qiskit import QuantumCircuit, QuantumRegister


def main_solver(instance, ansatz_tree, file, **kwargs):
    r"""
        This function solves Ax=b when Ansatz tree is known to us.
    """
    # Performing Hadamard test to calculate Q and r
    Q, r = calculate_Q_r(instance, ansatz_tree, **kwargs)
    file.writelines(['Q:', str(Q), '\n'])
    file.writelines(['r:', str(r), '\n'])
    # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
    loss, alphas = solve_combination_parameters(Q, r, which_opt='ADAM')
    print("loss:", loss)
    print("combination parameters are:", alphas)
    return loss, alphas


def __solve_and_expand(instance, ansatz_tree, **kwargs):
    loss, alphas = main_solver(instance, ansatz_tree, **kwargs)
    new_ansatz_tree = expand_ansatz_tree(instance, alphas, ansatz_tree, **kwargs)
    return loss, alphas, new_ansatz_tree


def main_prober(instance, backend=None, file=None, ITR=None, eps=None, **kwargs):
    r"""
        This function solves Ax=b when Ansatz tree is not known to us.
        Thus we use expansion algorithms to probe the solution space,
        including breadth-first search, gradient heuristic, etc.
    """
    n = instance.get_num_qubit()
    # For eigens simulator, each unitary gate is a list of Pauli strings.
    # For qiskit simulator, each unitary gate is a quantum circuit object in qiskit.
    if backend not in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ValueError("We do not allow the calls of this function in real ionq aria, since the "
                          "connection and execution takes time. Instead, we encourage the user to separate `submit`"
                          "process and `retrieve` process. Please try to build it using `main-solver`.")
    if eps is None:
        eps = 0.01

    # ansatz tree only contains identity at the beginning
    if backend == 'eigens':
        ansatz_tree = [[["I" for _ in range(n)]]]
    else:
        qr = QuantumRegister(n, 'q')
        id_cir = QuantumCircuit(qr)
        id_cir.id(qr)
        ansatz_tree = [id_cir]

    # main procedure
    LOSS = []
    Itr = []
    itr_count = 0
    loss = 1
    alphas = 0
    if ITR is None:
        while loss > eps:
            itr_count += 1
            Itr.append(itr_count)
            file.writelines(['Itr:', str(itr_count), '\n'])
            loss, alphas, ansatz_tree = __solve_and_expand(instance, ansatz_tree, file=file, backend=backend, **kwargs)
            # print("loss:", loss)
            LOSS.append(loss)
    else:
        while loss > eps:
            itr_count += 1
            if itr_count > ITR:
                break
            Itr.append(itr_count)
            file.writelines(['Itr:', str(itr_count), '\n'])
            loss, alphas, ansatz_tree = __solve_and_expand(instance, ansatz_tree, file=file, backend=backend, **kwargs)
            # print("loss:", loss)
            LOSS.append(loss)
    # print("combination parameters are:", alphas)

    return Itr, LOSS, ansatz_tree












































########################################################################################################################
#                                           CODES OF OLD VERSIONS
########################################################################################################################
# # Problem Setting
# # Generate the problem of solving a linear systems of equations.
# # Set the dimension of the coefficient matrix A on the left hand side of the equation.
# qubit_number = 5
# dim = 2 ** qubit_number
# # Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# # According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# # For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
# number_of_terms = 3
#
# # Total Budget of shots
# shots_total_budget = 10 ** 4
# # Expected error
# error = 0.1
#
# # Initialize the coefficient matrix
# A = CoeffMatrix(number_of_terms, dim, qubit_number)
# print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])
#
# # Generate A with the following way
# coeffs = [1, 0.5, 0.2]
# unitaries = [[['I', 'I', 'I', 'I', 'I']], [['X', 'Z', 'I', 'I', 'I']], [['X', 'I', 'Z', 'I', 'I']]]
# # Number of Hadamard tests in total: 636
# # So the total shot budget is: 636 * 11 =  6996
#
# # coeffs =  [0.358, 0.011, -0.919, -0.987]
# # unitaries = [[['I', 'Y', 'X', 'I', 'Y']], [['Z', 'X', 'Y', 'X', 'I']], [['X', 'X', 'Z', 'I', 'X']], [['Z', 'I', 'X', 'Z', 'Z']]]
#
# A.generate(which_form='Unitaries', given_unitaries=unitaries, given_coeffs=coeffs)
# print('Coefficients of the terms are:', coeffs)
# print('Decomposed unitaries are:', unitaries)
# B = sum([abs(coeff) for coeff in coeffs])
#
# # Values on the right hand side of the equation.
# b = array([1] + [0 for _ in range(dim - 1)])
# print('The vector on the right hand side of the equation is:', b)
#
# # 2. Initialization
# # Total iteration
# ITR = 3
# batch = shots_total_budget / (sum(list(i ** 2 for i in range(1, ITR + 1))))
# # Use a wise shot allocation method to display the shots
# Q_r_budgets = []
# loss_budgets = []
# gradient_budgets = []
# for itr in range(1, ITR + 1):
#     weight = itr ** 2
#     shot_budget = weight * batch
#
#     # Shot budget to calculate the matrix Q and vector r
#     Q_r_budget = int(0.4 * shot_budget)
#     Q_r_budgets.append(Q_r_budget)
#
#     # Shot budget to calculate the loss function
#     loss_budget = int(0.4 * shot_budget)
#     loss_budgets.append(loss_budget)
#
#     # Shot budget to calculate the gradient overlaps
#     gradient_budget = int(0.2 * shot_budget)
#     gradient_budgets.append(gradient_budget)
#
# Itr, loss_list_hadamard_qibo = main(backend='ionq', frugal=False)
# # Itr, loss_list_hadamard_not_frugal = main(backend='ionq', frugal=False)
# Itr, loss_list_eigens = main(backend='eigens', frugal=False)
#
# plt.title("CQS: Loss - Depth")
# plt.plot(Itr, [0 for _ in Itr], 'b--')
# plt.plot(Itr, loss_list_hadamard_qibo, 'g-', linewidth=2.5, label='Loss Function by Hadamard Tests with frugal method')
# # plt.plot(Itr, loss_list_hadamard_qibo, '-', color='black', linewidth=2.5, label='Loss Function by Hadamard Tests')
# plt.plot(Itr, loss_list_eigens, '-', color='red', linewidth=2.5, label='Loss Function by Matrix Multiplication')
# plt.legend()
# plt.xlabel("Depth")
# plt.ylabel("Loss")
# plt.show()

# xiufan_token = 'd383594cc2f9b3c7baf90b70b11f658c2df3ede2f534e571cc73d5e01f70e7a23620b4384594db7660ded289e864f137f3be56f05f39b9a113545ad90b11a302'
# IBMProvider.save_account(xiufan_token)

# Generate A with random Pauli matrices
# A.generate()
# Get the coefficients of the terms and the unitaries
# coeffs = A.get_coeff()
# unitaries = A.get_unitary()




# Analytical solution, we can get the matrix form of A
# mat = A.get_matrix()
# print('The coefficient matrix is:/n', mat)
# analy_x = linalg.inv(mat) @ b.reshape(dim, 1)
# print('The analytical solution of the linear systems of equations is:', analy_x)



# Choices of coefficients and unitaries
# coeffs = [0.5, -0.11037169750027553, 0.8186265152730448, -0.868868734050515]
# coeffs =  [0.3582017357918057, 0.011037169750027553, -0.9186265152730448, -0.9868868734050515]
# unitaries = [[['I', 'Y', 'X', 'I', 'Y']], [['Z', 'X', 'Y', 'X', 'I']], [['X', 'X', 'Z', 'I', 'X']], [['Z', 'I', 'X', 'Z', 'Z']]]
# coeffs = [1, 0.2, 0.2]
# coeffs = [1, 0.73, 0.2]
# coeffs = [1, 0.1, -0.5, 0.1]
# unitaries = [[['I', 'I', 'I', 'I', 'I']], [['I', 'X', 'I', 'X', 'I']], [['X', 'Z', 'I', 'I', 'Z']], [['Y', 'I', 'Z', 'Y', 'I']]]
# unitaries = [[['I', 'I', 'I', 'I', 'I']], [['X', 'Z', 'I', 'I', 'I']], [['X', 'I', 'Z', 'I', 'I']]]



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
#
#
# def Shot_Frugal_Main():
#
#     Itr = []
#     loss_list_hadamard_frugal = []
#     # At the begining, the ansatz tree only contains |b>, so we define it as [[['I', 'I']]]
#     ansatz_tree = [[['I' for _ in range(qubit_number)]]]
#
#     for itr in range(1, ITR + 1):
#         Itr.append(itr)
#         # Performing Hadamard test to calculate Q and r
#         backend = 'Hadamard'
#         Q, r = calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, backend=mtd, shots_budget=Q_r_budgets[itr - 1], frugal=True)
#
#         # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
#         vars = solve_combination_parameters(Q, r, which_opt=None)
#
#         # Calculate the regression loss function to test if it is in the error range
#         mtd = 'Hadamard'
#         loss = abs(real(calculate_loss_function(A, vars, ansatz_tree, mtd=mtd, shots_budget=loss_budgets[itr - 1], frugal=True)))
#         print('Tree Depth:', itr, "Loss:", loss)
#         loss_list_hadamard_frugal.append(loss)
#         mtd = 'Hadamard'
#         anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_budget=gradient_budgets[itr-1], frugal=True)
#     return Itr, loss_list_hadamard_frugal
#
#
# def Shot_not_Frugal_Main():
#     Itr = []
#     loss_list_hadamard_frugal = []
#     # At the begining, the ansatz tree only contains |b>, so we define it as [[['I', 'I']]]
#     ansatz_tree = [[['I' for _ in range(qubit_number)]]]
#
#     for itr in range(1, ITR + 1):
#         Itr.append(itr)
#         # Performing Hadamard test to calculate Q and r
#         mtd = 'Hadamard'
#         Q, r = calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=mtd, shots_budget=Q_r_budgets[itr - 1], frugal=False)
#
#         # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
#         vars = solve_combination_parameters(Q, r, which_opt=None)
#
#         # Calculate the regression loss function to test if it is in the error range
#         mtd = 'Hadamard'
#         loss = abs(real(
#             calculate_loss_function(A, vars, ansatz_tree, mtd=mtd, shots_budget=loss_budgets[itr - 1], frugal=False)))
#         print('Tree Depth:', itr, "Loss:", loss)
#         loss_list_hadamard_frugal.append(loss)
#         mtd = 'Hadamard'
#         anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False,
#                                          shots_budget=gradient_budgets[itr - 1], frugal=False)
#     return Itr, loss_list_hadamard_frugal
#
# def Matrix_Multiplication_Main():
#
#
#     ansatz_tree = [[['I' for _ in range(qubit_number)]]]
#
#
#     Itr = []
#     loss_list_matrix = []
#     itr = 0
#     loss = 1
#     mtd = 'Eigens'
#     # if qubit_number <= 10:
#     #     A_mat = A.get_matrix()
#     #     con_num = linalg.norm(A_mat) * linalg.norm(linalg.inv(A_mat))
#     #     rig_guar_mea = 10 ** 5 * (B ** 4) * (number_of_terms ** 2) * (con_num ** 2) / ((error) ** 5)
#     #     print(rig_guar_mea)
#     # Record the loss information
#
#     # for itr in range(1, total_tree_depth + 1):
#     # while abs(loss) >= error:
#     for _ in range(ITR):
#
#         itr += 1
#         shot_budget = itr * batch
#         Itr.append(itr)
#         eg = "Regression loss of depth" + str(itr)
#
#         Q, r = calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=mtd, shots_budget=shot_budget, frugal=False)
#
#         # Estimate the property of Q and r
#         # print('B:', B)
#         # print((itr ** 3) * (B ** 4) * (number_of_terms ** 2))
#         # measurement_shots = (itr ** 3) * (B ** 4) * (number_of_terms ** 2) *\
#         #                     linalg.norm(Q) * (linalg.norm(linalg.inv(Q))) ** 2 * \
#         #                     (1 + linalg.norm(linalg.inv(Q / 2) @ (r / (-2))) ** 2) / error
#         # print('Proposition 2 guarantees that the measurement shots should be more than:', int(measurement_shots.item()) + 1)
#
#         # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
#         vars = solve_combination_parameters(Q, r, which_opt=None)
#         # vars = solve_combination_parameters(Q, r, which_opt=None)
#         # Calculate the regression loss function to test if it is in the error range
#         # loss = calculate_loss_function(A, vars, ansatz_tree, shots=shots)
#         # print('Tree Depth:', itr, "test:", test)
#
#         # loss = real(calculate_loss_function(A, vars, ansatz_tree, shots=shots))
#         # mtd = 'Eigens'
#         loss = real(calculate_loss_function(A, vars, ansatz_tree, mtd=mtd, shots_budget=shot_budget, frugal=False))
#         print('Tree Depth:', itr, "Loss:", loss)
#         loss_list_matrix.append(loss)
#
#         # print("Error between loss_estimated and loss_ideal is:", loss_error)
#         # print()
#
#         # if itr % 10 == 0:
#         #     print("tree depth:", itr, "  loss:", "%.4f" % loss)
#
#         # If the loss is in our error range, stop the iteration;
#         # if abs(loss) <= error:
#         #     print(f"Find the optimal solution of combination parameters \n {vars} \n and the Ansatz tree structure \n {ansatz_tree}")
#         #     break
#         #
#         # else:
#         #     anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_power=shots_power)
#
#         anstaz_tree = expand_ansatz_tree(A, vars, ansatz_tree, mtd=mtd, draw_tree=False, shots_budget=shot_budget, frugal=False)
#
#     return Itr, loss_list_matrix