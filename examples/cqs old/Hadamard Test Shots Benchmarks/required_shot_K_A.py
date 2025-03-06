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
    This file is to test if there is a heuristic way to keep the required shots at a certain level.
    Required shots: to arrive at an error smaller than 0.1.
"""



from hardware.Qibo.qibo_access import Hadamard_test
from cqs.calculation import H_mat, X_mat, I_mat, Z_mat, Y_mat, zero_state
from numpy import kron, conj, transpose, log10, sqrt, linalg, random
from cqs.object import CoeffMatrix
from cqs.calculation import get_unitary, calculate_Q_r_by_Hadamrd_test

import matplotlib.pyplot as plt

def three_qubits(begin_shots, end_shots, exp_std):

    Error = []
    Log_Error = []
    Error_Bar = []
    Log_Error_Bar = []
    Shots = []
    Real_shots = []
    shot_sample = 100
    U = [['Y', 'Z', 'Y'], ['Z', 'Y', 'Z'], ['H', 'X', 'H'], ['X', 'H', 'X']]

    # step = int((end_shots - begin_shots) / 1000)
    # for shot in range(begin_shots, end_shots, step):
    # for shot in [int(10 ** ((i / number) - 1)) for i in list(range(int(log10(begin_shots)) * number, number * int(log10(end_shots))))]:
    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True

    plt.title("Figure 1 (log - log): Hadamard Test for a 3-qubit Unitary")

    for x in range(begin_shots, end_shots):
        shot = 10 ** x
        Shots.append(x)
        Real_shots.append(shot)
        error_each_shot = []
        log_error_each_shot = []
        for j in range(shot_sample):
            real_obs = Hadamard_test(U, shots=shot)
            imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
            exp_obs = real_obs - imag_obs * 1j
            error = linalg.norm(exp_obs - exp_std)
            log_error = log10(error)
            error_each_shot.append(error)
            log_error_each_shot.append(log_error)
        plt.scatter([x for _ in range(shot_sample)], log_error_each_shot, marker='o', s=2, c='g')
        error_ave = sum(error_each_shot) / shot_sample
        Error.append(error_ave)
        log_error_ave = log10(error_ave)
        Log_Error.append(log_error_ave)
        standard_derivation = sqrt(sum([(error - error_ave) ** 2 for error in error_each_shot]) / shot_sample)
        Error_Bar.append(standard_derivation)
        log_standard_derivation = sqrt(sum([(log10(error) - log10(error_ave)) ** 2 for error in error_each_shot]) / shot_sample)
        Log_Error_Bar.append(log_standard_derivation)

    plt.plot(Shots, [0 for _ in Shots], 'k-',
             Shots, [-1 for _ in Shots], 'g--',
             Shots, [-2 for _ in Shots], 'y--',
             Shots, [-3 for _ in Shots], 'm--',
             Shots, [-4 for _ in Shots], 'c--',
             Shots, [-5 for _ in Shots], 'b--',
             )
    plt.axvline(x=1, c="k")
    plt.axhline(y=0, c="k")
    plt.plot(Shots, Log_Error, '-', linewidth=3, color='k', label='Error-Shots')
    print("Shots are: ", Real_shots)
    print("Average Error is:", Error)
    print("Standard Derivation is:", Error_Bar)

    plt.errorbar(Shots, Log_Error, Log_Error_Bar, ecolor='r', elinewidth=2, capsize=10, fmt='o', color='k')
    plt.fill_between(Shots, [Log_Error[i] - Log_Error_Bar[i] for i in range(len(Shots))],
                     [Log_Error[i] + Log_Error_Bar[i] for i in range(len(Shots))],
                     color='gray', alpha=0.2)
    plt.xlabel("Shots / 10^x")
    plt.ylabel("Error / 10^y")
    plt.legend()
    plt.show()

    # Error = []
    # Error_Bar = []
    # Shots = []
    # Real_shots = []
    # shot_sample = 100
    #
    # plt.title("Figure 2 : Hadamard Test for a 3-qubit Unitary")
    #
    # for x in range(begin_shots, end_shots):
    #     shot = 10 ** x
    #     Real_shots.append(x)
    #     error_each_shot = []
    #     for j in range(shot_sample):
    #         real_obs = Hadamard_test(U, shots=shot)
    #         imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
    #         exp_obs = real_obs - imag_obs * 1j
    #         error = linalg.norm(exp_obs - exp_std)
    #         error_each_shot.append(error)
    #     plt.scatter([x for _ in range(shot_sample)], error_each_shot, marker='o', s=2, c='g')
    #     error_ave = sum(error_each_shot) / shot_sample
    #     Error.append(error_ave)
    #     standard_derivation = sqrt(sum([(error - error_ave) ** 2 for error in error_each_shot]) / shot_sample)
    #     Error_Bar.append(standard_derivation)
    #
    # plt.plot(Real_shots, [0 for _ in Real_shots], 'k-',
    #          Real_shots, [0.1 for _ in Real_shots], 'g--',
    #          Real_shots, [0.01 for _ in Real_shots], 'y--',
    #          Real_shots, [0.001 for _ in Real_shots], 'm--',
    #          Real_shots, [0.0001 for _ in Real_shots], 'c--',
    #          Real_shots, [0.00001 for _ in Real_shots], 'b--',
    #          )
    # plt.axvline(x=0, c="k")
    # plt.axhline(y=0, c="k")
    # plt.plot(Real_shots, Error, '-', linewidth=3, color='k', label='Error-Shots')
    #
    # plt.errorbar(Real_shots, Error, Error_Bar, ecolor='r', elinewidth=2, capsize=10, fmt='o', color='k')
    # plt.fill_between(Real_shots, [Error[i] - Error_Bar[i] for i in range(len(Real_shots))],
    #                  [Error[i] + Error_Bar[i] for i in range(len(Real_shots))],
    #                  color='gray', alpha=0.2)
    # plt.xlabel("Shots")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.show()


def four_qubits():

    U = [['H', 'X', 'H', 'H'], ['Z', 'Y', 'Z', 'Y'], ['Y', 'Z', 'Y', 'X'], ['X', 'H', 'X', 'Z']]
    zeros = kron(kron(kron(zero_state(), zero_state()), zero_state()), zero_state()).reshape(-1, 1)

    U_mat = kron(kron(kron(X_mat(), H_mat()), X_mat()), Z_mat()) \
            @ kron(kron(kron(Y_mat(), Z_mat()), Y_mat()), X_mat()) \
            @ kron(kron(kron(Z_mat(), Y_mat()), Z_mat()), Y_mat()) \
            @ kron(kron(kron(H_mat(), X_mat()), H_mat()), H_mat())

    exp_std = (conj(transpose(zeros)) @ U_mat @ zeros).item()

    Error = []
    Shots = []
    shots = 5000
    for shot in range(1000, shots):
        real_obs = Hadamard_test(U, shots=shot)
        imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
        exp_obs = real_obs - imag_obs * 1j
        error = linalg.norm(exp_obs - exp_std)
        Shots.append(shot)
        Error.append(error)

    plt.title("Hadamard Test: Error - Shots")
    # plt.plot([log10(shot) for shot in Shots], [0 for _ in range(1000, shots)], 'b--', [log10(shot) for shot in Shots], [log10(0.1) for _ in range(1000, shots)], 'r--', [log10(shot) for shot in Shots],
    #          [log10(0.01) for _ in range(1000, shots)], 'y--', [log10(shot) for shot in Shots], [log10(0.001) for _ in range(1000, shots)], 'g--', [log10(shot) for shot in Shots], Error,
    #          '.')
    plt.xlabel("Shots")
    plt.ylabel("Error")
    plt.show()

# three_qubits()
# input 10^x

def plot_shot_and_error(TBT, NAME, COLORS, ansatz_tree, begin_shots, end_shots):

    for i in range(len(TBT)):
        MTX = TBT[i]
        Q_exp, r_exp = calculate_Q_r_by_Hadamrd_test(MTX, ansatz_tree, backend='Matrix', shots_power=4)
        name = NAME[i]
        color = COLORS[i]

        Error = []
        Log_Error = []
        Error_Bar = []
        Log_Error_Bar = []
        Shots = []
        Real_shots = []
        shot_sample = 10
        # step = int((end_shots - begin_shots) / 1000)
        # for shot in range(begin_shots, end_shots, step):
        # for shot in [int(10 ** ((i / number) - 1)) for i in list(range(int(log10(begin_shots)) * number, number * int(log10(end_shots))))]:
        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        # plt.rcParams["figure.autolayout"] = True

        plt.title("Figure 1: Required Measurements of Different Sampling Techniques (log - log)")

        for x in range(begin_shots, end_shots):
            shot = 10 ** x
            Shots.append(x)
            Real_shots.append(shot)
            Q_error_each_shot = []
            Q_log_error_each_shot = []
            r_error_each_shot = []
            r_log_error_each_shot = []
            for j in range(shot_sample):
                # real_obs = Hadamard_test(U, shots=shot)
                # imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
                # exp_obs = real_obs - imag_obs * 1j
                Q, r = calculate_Q_r_by_Hadamrd_test(MTX, ansatz_tree, backend='Hadamard', shots_power=x)
                Q_error = linalg.norm(Q - Q_exp)
                r_error = linalg.norm(r - r_exp)

                Q_log_error = log10(Q_error)
                Q_error_each_shot.append(Q_error)
                Q_log_error_each_shot.append(Q_log_error)

                r_log_error = log10(r_error)
                r_error_each_shot.append(r_error)
                r_log_error_each_shot.append(r_log_error)

            plt.scatter([x for _ in range(shot_sample)], Q_log_error_each_shot, marker='o', s=2, c='g')
            # print("Scatter Data:", [([x for _ in range(shot_sample)][l], Q_log_error_each_shot[l]) for l in range(shot_sample)])

            print("Scatter Data, x:", [x for _ in range(shot_sample)])
            print("Scatter Data, log_error:", Q_log_error_each_shot)
            print()

            # plt.scatter([x for _ in range(shot_sample)], r_log_error_each_shot, marker='*', s=2, c='r')

            error_ave = sum(Q_error_each_shot) / shot_sample
            Error.append(error_ave)
            log_error_ave = log10(error_ave)
            Log_Error.append(log_error_ave)
            standard_derivation = sqrt(sum([(error - error_ave) ** 2 for error in Q_error_each_shot]) / shot_sample)
            Error_Bar.append(standard_derivation)
            log_standard_derivation = sqrt(sum([(log10(error) - log10(error_ave)) ** 2 for error in Q_error_each_shot]) / shot_sample)
            Log_Error_Bar.append(log_standard_derivation)

        print("Figure Data, Shots:", Shots)
        print("Figure Data, Log Error:", Log_Error)
        print("Figure Data, Log Error Bar:", Log_Error_Bar)
        print()


        plt.plot(Shots, Log_Error, '-', linewidth=3, color=color, label=name)
        print("Real Shots: ", Real_shots)
        print("Real Average Error:", Error)
        print("Real Standard Derivation:", Error_Bar)
        print()

        plt.errorbar(Shots, Log_Error, Log_Error_Bar, ecolor='r', elinewidth=2, capsize=10, fmt='o', color='k')
        plt.fill_between(Shots, [Log_Error[i] - Log_Error_Bar[i] for i in range(len(Shots))],
                         [Log_Error[i] + Log_Error_Bar[i] for i in range(len(Shots))],
                         color='gray', alpha=0.2)

    plt.plot(Shots, [0 for _ in Shots], 'k-',
             Shots, [-1 for _ in Shots], 'r--',
             Shots, [-2 for _ in Shots], 'r--',
             # Shots, [-3 for _ in Shots], 'm--',
             # Shots, [-4 for _ in Shots], 'c--',
             # Shots, [-5 for _ in Shots], 'b--',
             )
    plt.axvline(x=1, c="k")
    plt.axhline(y=0, c="k")

    plt.xlabel("Shots / 10^x")
    plt.ylabel("Error / 10^y")
    plt.legend()
    plt.show()

    # Error = []
    # Error_Bar = []
    # Shots = []
    # Real_shots = []
    # shot_sample = 100
    #
    # plt.title("Figure 2 : Hadamard Test for a 3-qubit Unitary")
    #
    # for x in range(begin_shots, end_shots):
    #     shot = 10 ** x
    #     Real_shots.append(x)
    #     error_each_shot = []
    #     for j in range(shot_sample):
    #         real_obs = Hadamard_test(U, shots=shot)
    #         imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
    #         exp_obs = real_obs - imag_obs * 1j
    #         error = linalg.norm(exp_obs - exp_std)
    #         error_each_shot.append(error)
    #     plt.scatter([x for _ in range(shot_sample)], error_each_shot, marker='o', s=2, c='g')
    #     error_ave = sum(error_each_shot) / shot_sample
    #     Error.append(error_ave)
    #     standard_derivation = sqrt(sum([(error - error_ave) ** 2 for error in error_each_shot]) / shot_sample)
    #     Error_Bar.append(standard_derivation)
    #
    # plt.plot(Real_shots, [0 for _ in Real_shots], 'k-',
    #          Real_shots, [0.1 for _ in Real_shots], 'g--',
    #          Real_shots, [0.01 for _ in Real_shots], 'y--',
    #          Real_shots, [0.001 for _ in Real_shots], 'm--',
    #          Real_shots, [0.0001 for _ in Real_shots], 'c--',
    #          Real_shots, [0.00001 for _ in Real_shots], 'b--',
    #          )
    # plt.axvline(x=0, c="k")
    # plt.axhline(y=0, c="k")
    # plt.plot(Real_shots, Error, '-', linewidth=3, color='k', label='Error-Shots')
    #
    # plt.errorbar(Real_shots, Error, Error_Bar, ecolor='r', elinewidth=2, capsize=10, fmt='o', color='k')
    # plt.fill_between(Real_shots, [Error[i] - Error_Bar[i] for i in range(len(Real_shots))],
    #                  [Error[i] + Error_Bar[i] for i in range(len(Real_shots))],
    #                  color='gray', alpha=0.2)
    # plt.xlabel("Shots")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.show()



# Generate the problem of solving a linear systems of equations.
# Set the dimension of the coefficient matrix A on the left hand side of the equation.
qubit_number = 9

dim = 2 ** qubit_number
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 10
# shots_power = 3

# total_tree_depth = 50
error = 0.1

# backend = 'Matrix'
backend = 'Hadamard'
# backend = 'Eigens'

# Initialize the coefficient matrix
A = CoeffMatrix(number_of_terms, dim, qubit_number)
# print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

# A_coeffs = [(random.rand() * 2) for _ in range(self.__term_number)]
A.generate()
A_coeffs = A.get_coeff()
unitaries = A.get_unitary()

B_coeffs = [(random.rand() * 2 - 1) for _ in range(number_of_terms)]
B = CoeffMatrix(number_of_terms, dim, qubit_number)
B.generate(which_form='Unitaries', given_coeffs=B_coeffs, given_unitaries=unitaries)


C_coeffs = [0.5 for _ in range(int(number_of_terms / 2))] + [-0.5 for _ in range(int(number_of_terms / 2))]
if len(C_coeffs) != number_of_terms:
    C_coeffs += [random.choice([-0.5, 0.5])]
C = CoeffMatrix(number_of_terms, dim, qubit_number)
C.generate(which_form='Unitaries', given_coeffs=C_coeffs, given_unitaries=unitaries)

D_coeffs = random.normal(0, 2, number_of_terms)
D = CoeffMatrix(number_of_terms, dim, qubit_number)
D.generate(which_form='Unitaries', given_coeffs=D_coeffs, given_unitaries=unitaries)

print("Experiment Data, Decomposed Unitaries:", unitaries)
print()
print("Experiment Data, A Coefficients:", A_coeffs)
print("Experiment Data, B Coefficients:", B_coeffs)
print("Experiment Data, C Coefficients:", C_coeffs)
print("Experiment Data, D Coefficients:", D_coeffs)
print()
# Generate A with Pauli matrices

# Generate A with other forms (Haar matrices)
# A.generate('Haar')

# Get the coefficients of the terms and the unitaries
ansatz_tree = [[['I' for _ in range(qubit_number)]]]

TBT = [A, B, C, D]
NAMES = ['Uniform Sampling: from [0, 2]', 'Uniform Sampling: from [-1, 1]',
         'Binary Sampling: from {-1, 1}', 'Normal Sampling: mu=0, sigma=2']
COLORS = ['blue', 'green', 'orange', 'purple']

plot_shot_and_error(TBT, NAMES, COLORS, ansatz_tree, 1, 6)









#
#
#
# U = [['Y', 'Z', 'Y'], ['Z', 'Y', 'Z'], ['H', 'X', 'H'], ['X', 'H', 'X']]
# zeros = kron(kron(zero_state(), zero_state()), zero_state()).reshape(-1, 1)
#
# U_mat = kron(kron(X_mat(), H_mat()), X_mat()) \
#         @ kron(kron(H_mat(), X_mat()), H_mat())\
#         @ kron(kron(Z_mat(), Y_mat()), Z_mat()) \
#         @ kron(kron(Y_mat(), Z_mat()), Y_mat())
# exp_std = (conj(transpose(zeros)) @ U_mat @ zeros).item()
#
# three_qubits(1, 5, exp_std)
#
#


























