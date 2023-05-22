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
from cqs_module.calculator import H_mat, X_mat, I_mat, Z_mat, Y_mat, zero_state
from numpy import kron, conj, transpose, log10, sqrt, linalg, random
from generator import CoeffMatrix
from cqs_module.calculator import get_unitary, calculate_Q_r_by_Hadamrd_test

import matplotlib.pyplot as plt


sampling_methods = ['Uniform Sampling in [0, 2]',
                    'Uniform Sampling in [-1, 1]',
                    'Binary Sampling from {-1, 1}',
                    'Normal Sampling with u =0, sigma = 2']
COLORS = ['blue', 'green', 'orange', 'purple']

shot_sample = 20
K_A_sample = 10

def plot_shots_and_K_A(start_K_A, end_K_A, error_bound):
    K_A_list = list(range(start_K_A, end_K_A))

    data_plot = {}

    for K_A in K_A_list:
        Required_shots_tot = [[] for _ in range(4)]
        for _ in range(K_A_sample):
            # Generate the problem of solving a linear systems of equations.
            # Set the dimension of the coefficient matrix A on the left hand side of the equation.
            qubit_number = K_A

            dim = 2 ** qubit_number
            # Set the number of terms of the coefficient matrix A on the left hand side of the equation.
            # According to assumption 1, the matrix A has the form of linear combination of known unitaries.
            # For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
            # shots_power = 3


            # Initialize the coefficient matrix
            A = CoeffMatrix(K_A, dim, qubit_number)
            # print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

            A.generate()
            unitaries = A.get_unitary()
            ansatz_tree = [[['I' for _ in range(qubit_number)]]]

            for i in range(4):
                sam_mtd = sampling_methods[i]

                # Uniform Sampling in [0, 2]
                if sam_mtd == 'Uniform Sampling in [0, 2]':
                    coeffs = [(random.rand() * 2) for _ in range(K_A)]

                # Uniform Sampling in [-1, 1]
                elif sam_mtd == 'Uniform Sampling in [-1, 1]':
                    coeffs = [(random.rand() * 2 - 1) for _ in range(K_A)]

                # Binary Sampling from {-1, 1}
                elif sam_mtd == 'Binary Sampling from {-1, 1}':
                    coeffs = [1 for _ in range(int(K_A / 2))] + [-1 for _ in range(int(K_A / 2))]
                    if len(coeffs) != K_A:
                        coeffs += [random.choice([-1, 1])]

                # Normal Sampling with u =0, sigma = 2
                elif sam_mtd == 'Normal Sampling with u =0, sigma = 2':
                    coeffs = random.normal(0, 2, K_A)

                else:
                    return ValueError

                B = CoeffMatrix(K_A, dim, qubit_number)
                B.generate(which_form='Unitaries', given_coeffs=coeffs, given_unitaries=unitaries)

                # print("Number of Decomposition Terms:", K_A)
                # print("Experiment Data, Decomposed Unitaries:", unitaries)
                # print("Experiment Data, " + sam_mtd + ' Coefficients :', coeffs)
                # print()

                Q_exp, r_exp = calculate_Q_r_by_Hadamrd_test(B, ansatz_tree, mtd='Matrix', shots_power=4)
                for x in range(6, 100):
                    Q_error_each_shot =[]
                    r_error_each_shot = []
                    for j in range(shot_sample):
                        Q, r = calculate_Q_r_by_Hadamrd_test(B, ansatz_tree, mtd='Hadamard', shots_power=x/3)
                        Q_error = linalg.norm(Q - Q_exp)
                        r_error = linalg.norm(r - r_exp)
                        Q_error_each_shot.append(Q_error)
                        r_error_each_shot.append(r_error)
                    error_ave = sum(Q_error_each_shot) / shot_sample
                    if error_ave <= error_bound:
                        required_shot = int(10 ** (x/3))
                        Required_shots_tot[i].append(required_shot)
                        plt.scatter(K_A, int(log10(required_shot)), marker='o', s=2, c='g')
                        # print("Required shots for", K_A, "terms is:", required_shot)
                        break
                    else:
                        if x <= 23:
                            continue
                        else:
                            required_shot = int(10 ** (24 / 3))
                            Required_shots_tot[i].append(required_shot)
                            plt.scatter(K_A, int(log10(required_shot)), marker='o', s=2, c='g')
                            break
        data_plot[K_A] = Required_shots_tot
    return data_plot

error_bound = 0.1
data_plot = plot_shots_and_K_A(1, 11, error_bound)
print("Required shots for each K_A with different samplings:", data_plot)
plt.title("Figure 2: log10(Required Shots) with Different Sampling Methods with error < " + str(error_bound))
data_ave = {K_A: [sum(data_plot[K_A][j]) / len(data_plot[K_A][j]) for j in range(4)] for K_A in list(data_plot.keys())}


for j in range(4):
    smp_mtd = sampling_methods[j]
    clr = COLORS[j]
    x_ = list(data_plot.keys())
    y_ = [log10(data_ave[K_A][j]) for K_A in x_]
    error_bar = [sqrt(sum([(log10(data_plot[x_[x]][j][k]) - y_[x_[x] - 1]) ** 2 for k in range(K_A_sample)]) / K_A_sample) for x in range(len(x_))]

    plt.plot(x_, y_, '-', color=clr, label=smp_mtd)
    plt.errorbar(x_, y_, error_bar, ecolor='r', elinewidth=2, capsize=10, fmt='o', color='k')
    plt.fill_between(x_, [y_[i] - error_bar[i] for i in range(len(x_))],
                     [y_[i] + error_bar[i] for i in range(len(x_))],
                     color='gray', alpha=0.2)
plt.xlabel("K_A / x")
plt.ylabel("Shots / 10^y")
plt.axvline(x=1, c="k")
plt.axhline(y=0, c="k")
plt.legend()
plt.show()


















