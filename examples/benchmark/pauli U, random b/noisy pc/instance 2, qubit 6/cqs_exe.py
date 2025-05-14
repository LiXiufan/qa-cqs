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
DEVICES = "aws-ionq-aria1"
IONQ_NOISE_LEVEL = 0.01
NOISE_LEVEL = IONQ_NOISE_LEVEL

Q_noiseless = array([[ 3.98710000e+01, -1.49624972e-01, -1.20480736e-14, -1.96917482e-15,
  -2.06790141e-16, -4.67625938e-17,  2.65927680e+01,  2.22044605e-16],
 [-1.49624972e-01,  3.98710000e+01, -3.44724249e-17, -1.03899112e-14,
   4.67625938e-17, -5.81934501e-16, -0.00000000e+00,  2.65927680e+01],
 [-1.20480736e-14, -3.44724249e-17,  3.98710000e+01,  1.49624972e-01,
  -2.65927680e+01,  0.00000000e+00,  2.87677659e-15,  1.66237024e-15],
 [-1.96917482e-15, -1.03899112e-14,  1.49624972e-01,  3.98710000e+01,
  -2.22044605e-16, -2.65927680e+01, -1.66237024e-15,  2.15455986e-15],
 [ 2.06790141e-16,  4.67625938e-17, -2.65927680e+01, -2.22044605e-16,
   3.98710000e+01, -1.49624972e-01, -1.20480736e-14, -1.96917482e-15],
 [-4.67625938e-17,  5.81934501e-16,  0.00000000e+00, -2.65927680e+01,
  -1.49624972e-01,  3.98710000e+01, -3.44724249e-17, -1.03899112e-14],
 [ 2.65927680e+01, -0.00000000e+00, -2.87677659e-15, -1.66237024e-15,
  -1.20480736e-14, -3.44724249e-17,  3.98710000e+01,  1.49624972e-01],
 [ 2.22044605e-16,  2.65927680e+01,  1.66237024e-15, -2.15455986e-15,
  -1.96917482e-15, -1.03899112e-14,  1.49624972e-01,  3.98710000e+01]])
r_noiseless = array([[ 0.00000000e+00],
 [ 3.97000000e+00],
 [-6.49480469e-17],
 [-1.80000000e-01],
 [ 0.00000000e+00],
 [ 0.00000000e+00],
 [ 6.49480469e-17],
 [-3.34921511e+00]])



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


def calculate_every_loss(Q, r, tree_depth):
    ALPHA = []
    LOSS = []
    LOSS_TRUE = []
    for depth in range(tree_depth):
        # Define the four sectors (quadrants)
        q1 = Q[:depth + 1, :depth + 1]
        q2 = Q[:depth + 1, tree_depth:tree_depth + depth + 1]
        q3 = Q[tree_depth:tree_depth + depth + 1, :depth + 1]
        q4 = Q[tree_depth:tree_depth + depth + 1, tree_depth:tree_depth + depth + 1]

        # Stack them back together
        top = hstack((q1, q2))
        bottom = hstack((q3, q4))
        Q_tem = vstack((top, bottom))

        r1 = r[:depth + 1]
        r2 = r[tree_depth:tree_depth + depth + 1]

        r_tem = vstack((r1, r2))
        loss, alpha = solve_combination_parameters(Q_tem, r_tem, which_opt='ADAM', reg=1)
        LOSS += [loss]
        ALPHA += [alpha]
        LOSS_TRUE += [find_true_loss_function(alpha, tree_depth)]
    return LOSS, LOSS_TRUE, ALPHA



with open('6_qubit_data_generation_matrix_A.csv', 'r', newline='') as csvfile:
    file_name_noiseless = 'instance_2995_result_noiseless.txt'
    file_name_noisy = 'instance_2995_result_noisy.txt'

    data_b=read_csv_b(6)
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i, row in enumerate(reader):
        if i == 2995:
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
            ub = qasm3.loads(data_b.iloc[i].qasm)

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
            Q_noisy, r_noisy = calculate_Q_r(instance, ansatz_tree, backend='qiskit-noisy', device=DEVICES,
                                             shots=0, optimization_level=2,
                                             noise_level_two_qubit=NOISE_LEVEL,
                                             noise_level_one_qubit=None,
                                             readout_error=None)

            # Create DataFrame
            Q_noisy_pd = pd.DataFrame(Q_noisy)
            r_noisy_pd = pd.DataFrame(r_noisy)
            # Save to CSV
            noisy_simulation_Q_csv_filename = "noisy_simulation_Q.csv"
            noisy_simulation_r_csv_filename = "noisy_simulation_r.csv"
            Q_noisy_pd.to_csv(noisy_simulation_Q_csv_filename, index=False)
            r_noisy_pd.to_csv(noisy_simulation_r_csv_filename, index=False)

            LOSS, LOSS_TRUE, ALPHA = calculate_every_loss(Q_noisy, r_noisy, tree_depth)
            LOSS = pd.DataFrame(LOSS)
            LOSS_TRUE = pd.DataFrame(LOSS_TRUE)
            ALPHA = pd.DataFrame(ALPHA)

            # Save to CSV
            noisy_simulation_loss_csv_filename = "noisy_simulation_loss.csv"
            noisy_simulation_true_loss_csv_filename = "noisy_simulation_true_loss.csv"
            noisy_simulation_alpha_csv_filename = "noisy_simulation_alpha.csv"

            LOSS.to_csv(noisy_simulation_loss_csv_filename, index=False)
            LOSS_TRUE.to_csv(noisy_simulation_true_loss_csv_filename, index=False)
            ALPHA.to_csv(noisy_simulation_alpha_csv_filename, index=False)



