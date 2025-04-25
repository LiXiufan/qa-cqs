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
from instances_b.reader_b import read_csv_b
from cqs.object import Instance
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.random import random_circuit

from examples.benchmark.cqs_main import main_prober, main_solver

from cqs.remote.calculation import submit_all_inner_products_in_V_dagger_V, submit_all_inner_products_in_q

# IONQ device two qubit gate fidelity: 98.510% = 0.9851
# IQM device two qubit gate fidelity: 99.163% = 0.9916
# IQM device readout error: 97.325% = 0.97325
DEVICES = ["aws-ionq-aria1", "aws-iqm-garnet"]
IONQ_NOISE_LEVEL = [0.005, 0.01, 0.015, 0.02, 0.025]
IQM_NOISE_LEVEL = [0.002, 0.005, 0.008, 0.011, 0.014]
NOISE_LEVEL = [IONQ_NOISE_LEVEL, IQM_NOISE_LEVEL]

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
                    loss, alphas = main_solver(instance, ansatz_tree, file_noisy, backend='qiskit-noisy', device=device,
                                shots=0, optimization_level=2,
                                noise_level_two_qubit=l,
                                noise_level_one_qubit=None,
                                readout_error=None)
                    file_noisy.writelines(['device is:', device, '\n'])
                    file_noisy.writelines(['noise level is:', str(l), '\n'])
                    file_noisy.writelines(['loss is:', str(loss), '\n'])
                    file_noisy.writelines(['alphas are:', str(alphas), '\n'])
                    file_noisy.writelines(['\n'])

            file_noisy.close()



