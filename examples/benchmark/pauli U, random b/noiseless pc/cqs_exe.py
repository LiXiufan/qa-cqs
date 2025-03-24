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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit

from transpiler.transpile import transpile_circuit
from examples.benchmark.cqs_main import main_prober


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

ITR = 3
data_file_name = '3_qubit_data_generation_matrix_A.csv'

with open(data_file_name, 'r', newline='') as csvfile:
    data_b=read_csv_b(3)
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i, row in enumerate(reader):
        if 2 > i > 0:
            file_name_noiseless = data_file_name + '_instance_' + str(i) + '_result_noiseless.txt'
            file_noiseless = open(file_name_noiseless, "a")
            row_clean = [j for j in ''.join(row).split('"') if j != ',']
            nLc = row_clean[0].split(',')
            n = int(nLc[0])
            print("qubit number is:", n)
            file_noiseless.writelines(['qubit number is:', str(n), '\n'])
            L = int(nLc[1])
            print("term number is:", L)
            file_noiseless.writelines(['term number is:', str(L), '\n'])
            kappa = float(nLc[2])
            print('condition number is', kappa)
            file_noiseless.writelines(['condition number is:', str(kappa), '\n'])
            pauli_strings = [__num_to_pauli_list(l) for l in eval(row_clean[1])]
            print('Pauli strings are:', pauli_strings)
            file_noiseless.writelines(['Pauli strings are:', str(pauli_strings), '\n'])
            pauli_circuits = [__num_to_pauli_circuit(l) for l in eval(row_clean[1])]
            coeffs = [float(i) for i in eval(row_clean[2])]
            print('coefficients are:', coeffs)
            file_noiseless.writelines(['Coefficients of the terms are:', str(coeffs), '\n'])
            print()

            # circuit depth d
            d = 3
            ub = qasm3.loads(data_b.iloc[i].qasm)#random_circuit(num_qubits=3, max_operands=2, depth=3, measure=False)
            print('Ub is given by:', data_b.iloc[i].b)
            print(ub)
            # generate instance
            instance = Instance(n, L, kappa)
            instance.generate(given_coeffs=coeffs, given_unitaries=pauli_circuits, given_ub=ub)
            Itr, LOSS, ansatz_tree = main_prober(instance, backend='qiskit-noiseless', file=file_noiseless, ITR=ITR, shots=0, optimization_level=2)
            print('Iterations are:', Itr)
            file_noiseless.writelines(['Iterations are:', str(Itr), '\n'])
            print('Losses are:', LOSS)
            file_noiseless.writelines(['Losses are:', str(LOSS), '\n'])
            file_noiseless.close()
            # matrix = instance.get_matrix()
            # print("The first example returns with a matrix:")
            # print(matrix)
            # print()














