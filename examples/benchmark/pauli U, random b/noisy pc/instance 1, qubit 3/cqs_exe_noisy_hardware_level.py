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

Q_noiseless = array([[ 1.00066000e+01, -4.10838030e-17, -5.57520000e+00,  5.89528426e-18,
   6.90980606e-16,  4.44089210e-16, -1.37486689e-15, -4.44089210e-16],
 [-4.10838030e-17,  1.00066000e+01, -6.68964883e-16, -5.57520000e+00,
  -4.44089210e-16,  8.48487947e-16, -4.44089210e-16, -1.13982712e-15],
 [-5.57520000e+00, -6.68964883e-16,  1.00066000e+01, -1.96792582e-16,
   1.37486689e-15,  4.44089210e-16,  8.76299033e-17,  1.02464703e-15],
 [ 5.89528426e-18, -5.57520000e+00, -1.96792582e-16,  1.00066000e+01,
   4.44089210e-16,  1.13982712e-15, -1.02464703e-15, -6.90980606e-16],
 [-6.90980606e-16, -4.44089210e-16,  1.37486689e-15,  4.44089210e-16,
   1.00066000e+01, -4.10838030e-17, -5.57520000e+00,  5.89528426e-18],
 [ 4.44089210e-16, -8.48487947e-16,  4.44089210e-16,  1.13982712e-15,
  -4.10838030e-17,  1.00066000e+01, -6.68964883e-16, -5.57520000e+00],
 [-1.37486689e-15, -4.44089210e-16, -8.76299033e-17, -1.02464703e-15,
  -5.57520000e+00, -6.68964883e-16,  1.00066000e+01, -1.96792582e-16],
 [-4.44089210e-16, -1.13982712e-15,  1.02464703e-15,  6.90980606e-16,
   5.89528426e-18, -5.57520000e+00, -1.96792582e-16,  1.00066000e+01]])
r_noiseless = array([[ 8.88178420e-18],
 [ 2.76000000e+00],
 [ 1.94844141e-16],
 [-1.01000000e+00],
 [ 0.00000000e+00],
 [-3.06421555e-16],
 [-1.17000000e+00],
 [ 9.71445147e-17]])



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



with open('3_qubit_data_generation_matrix_A.csv', 'r', newline='') as csvfile:
    file_name_noiseless = 'instance_1_result_noiseless.txt'
    file_name_noisy = 'instance_1_result_noisy.txt'

    data_b=read_csv_b(3)
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i, row in enumerate(reader):
        if i == 1:
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



