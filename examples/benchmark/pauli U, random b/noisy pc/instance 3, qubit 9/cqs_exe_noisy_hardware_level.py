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
    This file is used for benchmarking a larger number of instances_A.
"""

import csv
import qiskit.qasm3 as qasm3
import pandas as pd
from numpy import array, real, imag
from torch import Tensor, matmul
from numpy import vstack, hstack
from instances_b.reader_b import read_csv_b
from cqs.object import Instance
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.random import random_circuit
from cqs.local.calculation import calculate_Q_r
from cqs.optimization import solve_combination_parameters
from examples.benchmark.cqs_main import main_prober, main_solver

from cqs.remote.calculation import submit_all_inner_products_in_V_dagger_V, submit_all_inner_products_in_q

# IONQ device two qubit gate fidelity: 98.510% = 0.9851
# IQM device two qubit gate fidelity: 99.163% = 0.9916
# IQM device readout error: 97.325% = 0.97325
DEVICES = ["aws-ionq-aria1", "aws-iqm-garnet"]
IONQ_NOISE_LEVEL = [0.015]
IQM_NOISE_LEVEL = [0.0084]
NOISE_LEVEL = [IONQ_NOISE_LEVEL, IQM_NOISE_LEVEL]

Q_noiseless = array([[4.11260000e+00,  2.66453526e-19,  4.44089210e-16,  2.60902411e-19,
  -4.11260000e+00, -4.97935027e-17, -6.93889390e-17, -4.36650716e-17,
  -4.11260000e+00,  6.45483667e-17,  2.78999046e-16,  2.77999845e-17,
  -2.09310490e-03,  3.43836071e-17, -2.00197914e-15,  4.97935027e-17,
   3.14082094e-17,  4.36650716e-17, -5.03894704e-15, -1.47104551e-18],
 [ 2.66453526e-19,  4.11260000e+00,  2.45115039e-16, -3.33066907e-16,
   2.16493490e-18, -8.88178420e-16, -5.78204151e-17, -6.66133815e-16,
  -2.66453526e-19, -3.46560000e+00, -2.77999845e-17,  1.59899871e-16,
  -4.91495733e-17, -2.09310490e-03, -0.00000000e+00, -9.02651488e-03,
  -1.74660286e-16, -9.02651488e-03, -2.36255460e-17, -4.78259099e-15],
 [ 4.44089210e-16,  2.45115039e-16,  4.11260000e+00, -1.09112719e-16,
   1.94289029e-16, -1.03034248e-16, -6.66133815e-16,  9.84878845e-17,
   1.66533454e-16,  6.03850303e-17,  2.09310490e-03,  4.91495733e-17,
  -2.76054180e-16,  7.13540338e-17, -2.09310490e-03, -4.02178291e-17,
  -9.02651488e-03, -3.56714658e-17, -2.09310490e-03,  5.87974114e-17],
 [ 2.60902411e-19, -3.33066907e-16, -1.09112719e-16,  4.11260000e+00,
   3.81694676e-17,  6.93889390e-17, -4.36650716e-17,  5.55111512e-17,
  -1.08640874e-16,  3.64800000e-01, -3.43836071e-17,  2.09310490e-03,
  -7.13540338e-17,  7.77491960e-16, -1.27142741e-16, -6.12843110e-18,
   6.12843110e-18, -4.97935027e-17,  1.03778097e-16, -6.07780493e-16],
 [-4.11260000e+00,  2.16493490e-18,  1.94289029e-16,  3.81694676e-17,
   4.11260000e+00, -6.12843110e-18, -1.80411242e-16,  0.00000000e+00,
   4.11260000e+00,  1.34120492e-16,  2.00197914e-15,  0.00000000e+00,
   2.09310490e-03,  1.27142741e-16,  7.68321518e-16, -4.36650716e-17,
   2.37476705e-17,  4.97935027e-17,  4.69252692e-15, -1.30473410e-16],
 [-4.97935027e-17, -8.88178420e-16, -1.03034248e-16,  6.93889390e-17,
  -6.12843110e-18,  4.11260000e+00, -1.54681823e-16,  4.11260000e+00,
   4.34552394e-16,  1.57320000e+00, -4.97935027e-17,  9.02651488e-03,
   4.02178291e-17,  6.12843110e-18,  4.36650716e-17,  2.50791055e-16,
  -6.70796751e-17,  1.85998716e-15, -8.09907696e-17,  9.16816623e-16],
 [-6.93889390e-17, -5.78204151e-17, -6.66133815e-16, -4.36650716e-17,
  -1.80411242e-16, -1.54681823e-16,  4.11260000e+00, -2.52237120e-16,
   1.52655666e-16, -7.22810700e-17, -3.14082094e-17,  1.74660286e-16,
   9.02651488e-03, -6.12843110e-18, -2.37476705e-17,  6.70796751e-17,
   2.28966013e-15,  7.35633776e-17,  1.04027897e-17,  6.20503648e-17],
 [-4.36650716e-17, -6.66133815e-16,  9.84878845e-17,  5.55111512e-17,
   0.00000000e+00,  4.11260000e+00, -2.52237120e-16,  4.11260000e+00,
  -2.56505928e-16,  1.57320000e+00, -4.36650716e-17,  9.02651488e-03,
   3.56714658e-17,  4.97935027e-17, -4.97935027e-17, -1.85998716e-15,
  -7.35633776e-17,  2.64893663e-16, -2.91511260e-16,  7.64238672e-16],
 [-4.11260000e+00, -2.66453526e-19,  1.66533454e-16, -1.08640874e-16,
   4.11260000e+00,  4.34552394e-16,  1.52655666e-16, -2.56505928e-16,
   4.11260000e+00, -9.87210313e-17,  5.03894704e-15,  2.36255460e-17,
   2.09310490e-03, -1.03778097e-16, -4.69252692e-15,  8.09907696e-17,
  -1.04027897e-17,  2.91511260e-16,  9.77634640e-16,  8.62698801e-17],
 [ 6.45483667e-17, -3.46560000e+00,  6.03850303e-17,  3.64800000e-01,
   1.34120492e-16,  1.57320000e+00, -7.22810700e-17,  1.57320000e+00,
  -9.87210313e-17,  4.11260000e+00,  1.47104551e-18,  4.78259099e-15,
  -5.87974114e-17,  6.07780493e-16,  1.30473410e-16, -9.16816623e-16,
  -6.20503648e-17, -7.64238672e-16, -8.62698801e-17,  3.28154171e-16],
 [-2.78999046e-16, -2.77999845e-17,  2.09310490e-03, -3.43836071e-17,
   2.00197914e-15, -4.97935027e-17, -3.14082094e-17, -4.36650716e-17,
   5.03894704e-15,  1.47104551e-18,  4.11260000e+00,  2.66453526e-19,
   4.44089210e-16,  2.60902411e-19, -4.11260000e+00, -4.97935027e-17,
  -6.93889390e-17, -4.36650716e-17, -4.11260000e+00,  6.45483667e-17],
 [ 2.77999845e-17, -1.59899871e-16,  4.91495733e-17,  2.09310490e-03,
   0.00000000e+00,  9.02651488e-03,  1.74660286e-16,  9.02651488e-03,
   2.36255460e-17,  4.78259099e-15,  2.66453526e-19,  4.11260000e+00,
   2.45115039e-16, -3.33066907e-16,  2.16493490e-18, -8.88178420e-16,
  -5.78204151e-17, -6.66133815e-16, -2.66453526e-19, -3.46560000e+00],
 [-2.09310490e-03, -4.91495733e-17,  2.76054180e-16, -7.13540338e-17,
   2.09310490e-03,  4.02178291e-17,  9.02651488e-03,  3.56714658e-17,
   2.09310490e-03, -5.87974114e-17,  4.44089210e-16,  2.45115039e-16,
   4.11260000e+00, -1.09112719e-16,  1.94289029e-16, -1.03034248e-16,
  -6.66133815e-16,  9.84878845e-17,  1.66533454e-16,  6.03850303e-17],
 [ 3.43836071e-17, -2.09310490e-03,  7.13540338e-17, -7.77491960e-16,
   1.27142741e-16,  6.12843110e-18, -6.12843110e-18,  4.97935027e-17,
  -1.03778097e-16,  6.07780493e-16,  2.60902411e-19, -3.33066907e-16,
  -1.09112719e-16,  4.11260000e+00,  3.81694676e-17,  6.93889390e-17,
  -4.36650716e-17,  5.55111512e-17, -1.08640874e-16, 3.64800000e-01],
 [-2.00197914e-15, -0.00000000e+00, -2.09310490e-03, -1.27142741e-16,
  -7.68321518e-16,  4.36650716e-17, -2.37476705e-17, -4.97935027e-17,
  -4.69252692e-15,  1.30473410e-16, -4.11260000e+00,  2.16493490e-18,
   1.94289029e-16,  3.81694676e-17,  4.11260000e+00, -6.12843110e-18,
  -1.80411242e-16,  0.00000000e+00,  4.11260000e+00,  1.34120492e-16],
 [ 4.97935027e-17, -9.02651488e-03, -4.02178291e-17, -6.12843110e-18,
  -4.36650716e-17, -2.50791055e-16,  6.70796751e-17, -1.85998716e-15,
   8.09907696e-17, -9.16816623e-16, -4.97935027e-17, -8.88178420e-16,
  -1.03034248e-16,  6.93889390e-17, -6.12843110e-18,  4.11260000e+00,
  -1.54681823e-16,  4.11260000e+00,  4.34552394e-16,  1.57320000e+00],
 [ 3.14082094e-17, -1.74660286e-16, -9.02651488e-03,  6.12843110e-18,
   2.37476705e-17, -6.70796751e-17, -2.28966013e-15, -7.35633776e-17,
  -1.04027897e-17, -6.20503648e-17, -6.93889390e-17, -5.78204151e-17,
  -6.66133815e-16, -4.36650716e-17, -1.80411242e-16, -1.54681823e-16,
   4.11260000e+00, -2.52237120e-16,  1.52655666e-16, -7.22810700e-17],
 [ 4.36650716e-17, -9.02651488e-03, -3.56714658e-17, -4.97935027e-17,
   4.97935027e-17,  1.85998716e-15,  7.35633776e-17, -2.64893663e-16,
   2.91511260e-16, -7.64238672e-16, -4.36650716e-17, -6.66133815e-16,
   9.84878845e-17,  5.55111512e-17,  0.00000000e+00,  4.11260000e+00,
  -2.52237120e-16,  4.11260000e+00, -2.56505928e-16,  1.57320000e+00],
 [-5.03894704e-15, -2.36255460e-17, -2.09310490e-03,  1.03778097e-16,
   4.69252692e-15, -8.09907696e-17,  1.04027897e-17, -2.91511260e-16,
  -9.77634640e-16, -8.62698801e-17, -4.11260000e+00, -2.66453526e-19,
   1.66533454e-16, -1.08640874e-16,  4.11260000e+00,  4.34552394e-16,
   1.52655666e-16, -2.56505928e-16,  4.11260000e+00, -9.87210313e-17],
 [-1.47104551e-18, -4.78259099e-15,  5.87974114e-17, -6.07780493e-16,
  -1.30473410e-16,  9.16816623e-16,  6.20503648e-17,  7.64238672e-16,
   8.62698801e-17, -3.28154171e-16,  6.45483667e-17, -3.46560000e+00,
   6.03850303e-17,  3.64800000e-01,  1.34120492e-16,  1.57320000e+00,
  -7.22810700e-17,  1.57320000e+00, -9.87210313e-17,  4.11260000e+00]])
r_noiseless = array([[-8.88178420e-18],
 [-1.52000000e+00],
 [-1.66533454e-18],
 [ 1.60000000e-01],
 [-8.88178420e-18],
 [ 6.90000000e-01],
 [ 0.00000000e+00],
 [ 6.90000000e-01],
 [ 0.00000000e+00],
 [ 1.14000000e+00],
 [ 0.00000000e+00],
 [-6.54095281e-03],
 [-1.66533454e-18],
 [ 3.55271368e-17],
 [-5.44009282e-17],
 [ 3.83026943e-16],
 [ 0.00000000e+00],
 [ 2.29816166e-16],
 [ 8.88178420e-18],
 [ 5.60662627e-16]])



def __num_to_pauli_list(num_list):
    paulis = ['I', 'X', 'Y', 'Z']
    pauli_list = [paulis[int(i)] for i in num_list]
    return pauli_list

def __add_Pauli_gate(qc, which_qubit, which_gate):
    if which_gate == 0:
        qc.id(which_qubit)
    elif which_gate == 1:
        qc.x(which_qubit)
    elif which_gate == 2:
        qc.y(which_qubit)
    elif which_gate == 3:
        qc.z(which_qubit)
    else:
        return ValueError("Not supported Pauli gate type.")

def __num_to_pauli_circuit(num_list):
    n = len(num_list)
    num_list = [int(i) for i in num_list]
    qr = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qr)
    for i in range(n):
        __add_Pauli_gate(qc, i, num_list[i])
    return qc

def create_random_circuit_in_native_gate(n, d):
    ub = random_circuit(num_qubits=n,max_operands=2, depth=d, measure=False)
    # ub = transpile_circuit(ub, device='Aria', optimization_level=2)
    return ub


def find_true_loss_function(alphas, tree_depth):
    x = vstack((real(alphas), imag(alphas))).reshape(-1, 1)
    depth = len(alphas) - 1
    # Define the four sectors (quadrants)
    q1 = Q_noiseless[:depth + 1, :depth + 1]
    q2 = Q_noiseless[:depth + 1, tree_depth:tree_depth + depth + 1]
    q3 = Q_noiseless[tree_depth:tree_depth + depth + 1, :depth + 1]
    q4 = Q_noiseless[tree_depth:tree_depth + depth + 1, tree_depth:tree_depth + depth + 1]

    # Stack them back together
    top = hstack((q1, q2))
    bottom = hstack((q3, q4))
    Q_tem = vstack((top, bottom))

    r1 = r_noiseless[:depth + 1]
    r2 = r_noiseless[tree_depth:tree_depth + depth + 1]

    r_tem = vstack((r1, r2)).reshape(-1, 1)
    xt = Tensor(x)
    Qt = Tensor(Q_tem) * 2
    rt = Tensor(r_tem) * (-2)
    return abs((0.5 * matmul(xt.T, matmul(Qt, xt)) + matmul(rt.T, xt) + 1).item())



with open('9_qubit_data_generation_matrix_A.csv', 'r', newline='') as csvfile:
    file_name_noiseless = 'instance_2021_result_noiseless.txt'
    file_name_noisy = 'instance_2021_result_noisy.txt'

    data_b=read_csv_b(9)
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i, row in enumerate(reader):
        if i == 2021:
            file_noiseless = open(file_name_noiseless, "a")
            row_clean = [j for j in ''.join(row).split('"') if j != ',']
            nLc = row_clean[0].split(',')
            n = int(nLc[0])
            L = int(nLc[1])
            kappa = float(nLc[2])
            pauli_strings = [__num_to_pauli_list(l) for l in eval(row_clean[1])]
            pauli_circuits = [__num_to_pauli_circuit(l) for l in eval(row_clean[1])]
            coeffs = [float(i) for i in eval(row_clean[2])]

            file_noiseless.writelines(['qubit number is:', str(n), '\n'])
            file_noiseless.writelines(['term number is:', str(L), '\n'])
            file_noiseless.writelines(['condition number is:', str(kappa), '\n'])
            file_noiseless.writelines(['Pauli strings are:', str(pauli_strings), '\n'])
            file_noiseless.writelines(['Coefficients of the terms are:', str(coeffs), '\n'])

            # circuit depth d
            ub = qasm3.loads(data_b.iloc[i].qasm)#random_circuit(num_qubits=3, max_operands=2, depth=3, measure=False)

            # generate instance
            instance = Instance(n, L, kappa)
            instance.generate(given_coeffs=coeffs, given_unitaries=pauli_circuits, given_ub=ub)
            Itr, LOSS, ansatz_tree = main_prober(instance, backend='qiskit-noiseless', file=file_noiseless, ITR=None, shots=0, optimization_level=2)
            # remove the last expanded gate
            ansatz_tree = [ansatz_tree[i] for i in range(len(ansatz_tree) - 1)]
            file_noiseless.writelines(['Iterations are:', str(Itr), '\n'])
            file_noiseless.writelines(['Losses are:', str(LOSS), '\n'])
            file_noiseless.close()

            tree_depth = len(ansatz_tree)

            # perform noisy simulation
            file_noisy = open(file_name_noisy, "a")
            file_noisy.writelines(['qubit number is:', str(n), '\n'])
            file_noisy.writelines(['term number is:', str(L), '\n'])
            file_noisy.writelines(['condition number is:', str(kappa), '\n'])
            file_noisy.writelines(['Pauli strings are:', str(pauli_strings), '\n'])
            file_noisy.writelines(['Coefficients of the terms are:', str(coeffs), '\n'])


            for j in range(2):
                device = DEVICES[j]
                noise_level_two_qubit = NOISE_LEVEL[j]
                for l in noise_level_two_qubit:
                    file_noisy.writelines(['device is:', device, '\n'])
                    file_noisy.writelines(['noise level is:', str(l), '\n'])
                    for d in range(tree_depth):
                        file_noisy.writelines(['depth:', str(d), '\n'])
                        ansatz_tree_d = ansatz_tree[:d + 1]
                        # Performing Hadamard test to calculate Q and r
                        Q_noisy, r_noisy = calculate_Q_r(instance, ansatz_tree_d, backend='qiskit-noisy', device=device,
                                    shots=0, optimization_level=2,
                                    noise_level_two_qubit=l,
                                    noise_level_one_qubit=None,
                                    readout_error=None)
                        file_noisy.writelines(['Q_noisy:', str(Q_noisy), '\n'])
                        file_noisy.writelines(['r_noisy:', str(r_noisy), '\n'])
                        # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
                        train_loss, alphas = solve_combination_parameters(Q_noisy, r_noisy, which_opt='ADAM')
                        file_noisy.writelines(['train loss:', str(train_loss), '\n'])
                        file_noisy.writelines(['alphas:', str(alphas), '\n'])

                        # calculate the true loss
                        true_loss = find_true_loss_function(alphas, d)
                        file_noisy.writelines(['true loss:', str(true_loss), '\n'])
                        file_noisy.writelines(['\n'])
            file_noisy.close()



