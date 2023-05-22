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
    Estimation of shots number for each Hadamard test.
"""



from hardware.Qibo.qibo_access import Hadamard_test
from cqs_module.calculator import H_mat, X_mat, I_mat, Z_mat, Y_mat, zero_state
from numpy import kron, conj, transpose, log10, sqrt
from numpy import linalg, array
from numpy import poly1d, polyfit, linspace, real, imag


import matplotlib.pyplot as plt
Exp = []
Exp_real = []
Exp_imag = []

Shots = []

M = 3
unbiased_number = 10

U = [['Y', 'Z', 'Y'], ['Z', 'Y', 'Z'], ['H', 'X', 'H'], ['X', 'H', 'X']]
zeros = kron(kron(zero_state(), zero_state()), zero_state()).reshape(-1, 1)

U_mat = kron(kron(X_mat(), H_mat()), X_mat()) \
        @ kron(kron(H_mat(), X_mat()), H_mat())\
        @ kron(kron(Z_mat(), Y_mat()), Z_mat()) \
        @ kron(kron(Y_mat(), Z_mat()), Y_mat())

exp_std = (conj(transpose(zeros)) @ (U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat + U_mat) @ zeros).item()




# real_exp_std = real(exp_std)


# Richardson Extrapolation

for m in range(M, 2, -1):
    shot = int(10 ** (m))
    unbiased_exp_real = []
    unbiased_exp_imag = []
    unbiased_exp = []

    for itr in range(unbiased_number):
        exp_obs = 0
        for k in range(14):
            real_obs_ = Hadamard_test(U, shots=shot)
            imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
            exp_obs += real_obs_ - imag_obs * 1j
        exp_obs_real = real(exp_obs)
        exp_obs_imag = imag(exp_obs)

        unbiased_exp_real.append(exp_obs_real)
        unbiased_exp_imag.append(exp_obs_imag)
        unbiased_exp.append(exp_obs)


    exp_obs_real_ave = sum(unbiased_exp_real) / unbiased_number
    unbiased_exp_real_half = [i for i in unbiased_exp_real if i >= exp_obs_real_ave]
    unbiased_exp_real_half_exp = sum(unbiased_exp_real_half) / len(unbiased_exp_real_half)
    Exp_real.append(unbiased_exp_real_half_exp)

    exp_obs_imag_ave = sum(unbiased_exp_imag) / unbiased_number
    unbiased_exp_imag_half = [i for i in unbiased_exp_imag if i >= exp_obs_imag_ave]
    unbiased_exp_imag_half_exp = sum(unbiased_exp_imag_half) / len(unbiased_exp_imag_half)
    Exp_imag.append(unbiased_exp_imag_half_exp)

    exp_obs_ave = sum(unbiased_exp) / unbiased_number
    unbiased_exp_half = [i for i in unbiased_exp if i >= exp_obs_ave]
    unbiased_exp_half_exp = sum(unbiased_exp_half) / len(unbiased_exp_half)
    Exp.append(unbiased_exp_half_exp)


    Shots.append(shot)

print('Error before mitigation.py:', [linalg.norm(i - exp_std) for i in Exp])


exp_mtg_real = 0
length = len(Shots)
for m in range(1, length+1):
    exp_obs_real = Exp_real[m - 1]
    shot_m = Shots[m - 1]
    multiplication = 1
    for k in range(1, length+1):
        if k == m:
            continue
        else:
            shot_k = Shots[k - 1]
            # multiplication *= (shot_m ** 0.5) / (shot_m ** 0.5 - shot_k ** 0.5)
            multiplication *= (shot_m) / (shot_m - shot_k)

    exp_mtg_real += exp_obs_real * multiplication


exp_mtg_imag = 0
length = len(Shots)
for m in range(1, length+1):
    exp_obs_imag = Exp_imag[m - 1]
    shot_m = Shots[m - 1]
    multiplication = 1
    for k in range(1, length+1):
        if k == m:
            continue
        else:
            shot_k = Shots[k - 1]
            # multiplication *= (shot_m ** 0.5) / (shot_m ** 0.5 - shot_k ** 0.5)
            multiplication *= (shot_m) / (shot_m - shot_k)

    exp_mtg_imag += exp_obs_imag * multiplication

exp_mtg = exp_mtg_real + exp_mtg_imag *1j

print('Error after zero-noise mitigation.py:', [linalg.norm(exp_mtg - exp_std)])




# Robust Regression with Polynomials
#
# for m in range(M, 6, -1):
#     shot = int(10 ** (m/3))
#     unbiased_exp = []
#     for itr in range(unbiased_number):
#         Shots.append(1 / (shot ** 0.5))
#         real_obs = Hadamard_test(U, shots=shot)
#         # imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
#         # exp_obs = real_obs - imag_obs * 1j
#         Exp.append(real_obs)
#     # exp_obs_ave = sum(unbiased_exp) / unbiased_number
#
# print('Error before mitigation.py:', [linalg.norm(i - real_exp_std) for i in Exp])
#
# model = poly1d(polyfit(Exp, Shots, 2))
# a, b, c = model.c
# x_min = - b / (2 * a)
# y_min = model(x_min)
# print('Error after mitigation.py is:', linalg.norm(y_min - real_exp_std))
# X = array([x_min])
# Y = array([y_min])
# plt.scatter(X, Y)
#
#
# polyline = linspace(-0.1, 0.1, 100)
# plt.scatter(Exp, Shots)
# plt.plot(polyline, model(polyline))
# plt.show()






#
#
#
# def three_qubits(begin_shots, end_shots, exp_std):
#
#     Error = []
#     Log_Error = []
#     Error_Bar = []
#     Log_Error_Bar = []
#     Shots = []
#     Real_shots = []
#     shot_sample = 100
#     # step = int((end_shots - begin_shots) / 1000)
#     # for shot in range(begin_shots, end_shots, step):
#     # for shot in [int(10 ** ((i / number) - 1)) for i in list(range(int(log10(begin_shots)) * number, number * int(log10(end_shots))))]:
#     # plt.rcParams["figure.figsize"] = [7.50, 3.50]
#     # plt.rcParams["figure.autolayout"] = True
#
#     plt.title("Figure 1 (log - log): Hadamard Test for a 3-qubit Unitary")
#
#     for x in range(begin_shots, end_shots):
#         shot = 10 ** x
#         Shots.append(x)
#         Real_shots.append(shot)
#         error_each_shot = []
#         log_error_each_shot = []
#         for j in range(shot_sample):
#             real_obs = Hadamard_test(U, shots=shot)
#             imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
#             exp_obs = real_obs - imag_obs * 1j
#             error = linalg.norm(exp_obs - exp_std)
#             log_error = log10(error)
#             error_each_shot.append(error)
#             log_error_each_shot.append(log_error)
#         plt.scatter([x for _ in range(shot_sample)], log_error_each_shot, marker='o', s=2, c='g')
#         error_ave = sum(error_each_shot) / shot_sample
#         Error.append(error_ave)
#         log_error_ave = log10(error_ave)
#         Log_Error.append(log_error_ave)
#         standard_derivation = sqrt(sum([(error - error_ave) ** 2 for error in error_each_shot]) / shot_sample)
#         Error_Bar.append(standard_derivation)
#         log_standard_derivation = sqrt(sum([(log10(error) - log10(error_ave)) ** 2 for error in error_each_shot]) / shot_sample)
#         Log_Error_Bar.append(log_standard_derivation)
#
#     plt.plot(Shots, [0 for _ in Shots], 'k-',
#              Shots, [-1 for _ in Shots], 'g--',
#              Shots, [-2 for _ in Shots], 'y--',
#              Shots, [-3 for _ in Shots], 'm--',
#              Shots, [-4 for _ in Shots], 'c--',
#              Shots, [-5 for _ in Shots], 'b--',
#              )
#     plt.axvline(x=1, c="k")
#     plt.axhline(y=0, c="k")
#     plt.plot(Shots, Log_Error, '-', linewidth=3, color='k', label='Error-Shots')
#     print("Shots are: ", Real_shots)
#     print("Average Error is:", Error)
#     print("Standard Derivation is:", Error_Bar)
#
#     plt.errorbar(Shots, Log_Error, Log_Error_Bar, ecolor='r', elinewidth=2, capsize=10, fmt='o', color='k')
#     plt.fill_between(Shots, [Log_Error[i] - Log_Error_Bar[i] for i in range(len(Shots))],
#                      [Log_Error[i] + Log_Error_Bar[i] for i in range(len(Shots))],
#                      color='gray', alpha=0.2)
#     plt.xlabel("Shots / 10^x")
#     plt.ylabel("Error / 10^y")
#     plt.legend()
#     plt.show()

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

#
# def four_qubits():
#
#     U = [['H', 'X', 'H', 'H'], ['Z', 'Y', 'Z', 'Y'], ['Y', 'Z', 'Y', 'X'], ['X', 'H', 'X', 'Z']]
#     zeros = kron(kron(kron(zero_state(), zero_state()), zero_state()), zero_state()).reshape(-1, 1)
#
#     U_mat = kron(kron(kron(X_mat(), H_mat()), X_mat()), Z_mat()) \
#             @ kron(kron(kron(Y_mat(), Z_mat()), Y_mat()), X_mat()) \
#             @ kron(kron(kron(Z_mat(), Y_mat()), Z_mat()), Y_mat()) \
#             @ kron(kron(kron(H_mat(), X_mat()), H_mat()), H_mat())
#
#     exp_std = (conj(transpose(zeros)) @ U_mat @ zeros).item()
#
#     Error = []
#     Shots = []
#     shots = 5000
#     for shot in range(1000, shots):
#         real_obs = Hadamard_test(U, shots=shot)
#         imag_obs = Hadamard_test(U, alpha=1j, shots=shot)
#         exp_obs = real_obs - imag_obs * 1j
#         error = linalg.norm(exp_obs - exp_std)
#         Shots.append(shot)
#         Error.append(error)
#
#     plt.title("Hadamard Test: Error - Shots")
#     # plt.plot([log10(shot) for shot in Shots], [0 for _ in range(1000, shots)], 'b--', [log10(shot) for shot in Shots], [log10(0.1) for _ in range(1000, shots)], 'r--', [log10(shot) for shot in Shots],
#     #          [log10(0.01) for _ in range(1000, shots)], 'y--', [log10(shot) for shot in Shots], [log10(0.001) for _ in range(1000, shots)], 'g--', [log10(shot) for shot in Shots], Error,
#     #          '.')
#     plt.xlabel("Shots")
#     plt.ylabel("Error")
#     plt.show()
#
# # three_qubits()
# # input 10^x
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
# three_qubits(1, 11, exp_std)
#
#
#
#
#
#






















