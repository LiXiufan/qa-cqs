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
    This file is used for calculating auxiliary systems Q, r;
    performing convex optimization and obtaining the hardware loss.
"""

import csv
import qiskit.qasm3 as qasm3
import pandas as pd
from instances_b.reader_b import read_csv_b
from cqs.object import Instance
from qiskit import QuantumCircuit, QuantumRegister
from numpy import array, zeros, conj, real, imag
from numpy import vstack, hstack
from collections import Counter

from torch import Tensor, matmul

from cqs.remote.calculation import reshape_to_Q_r
from cqs.optimization import solve_combination_parameters

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


def __calculate_exp_by_count(count):
    prob_result = {str(outcome): int(count[outcome]) for outcome in
                   count.keys()}
    count0 = []
    count1 = []
    for outcome in prob_result.keys():
        if outcome[0] == '0':
            count0.append(prob_result[outcome])
        else:
            count1.append(prob_result[outcome])
    shots = sum(count0) + sum(count1)
    if not count0:
        p0 = 0
        p1 = 1
    elif not count1:
        p0 = 1
        p1 = 0
    else:
        p0 = sum(count0) / shots
        # Error mitigation
        # p0 = (p0 - 0.0048) / (1 - 2 * 0.0048)
        p1 = 1 - p0
    return p0 - p1

def calculate_V_dagger_V_from_counts(instance, tree_depth, counts):
    r"""
        Estimate all independent inner products that appear in matrix `V_dagger_V`.
        Note that we only estimate all upper triangular elements and all diagonal elements.
        For lower elements, since the matrix `V_dagger_V` is symmetrical, we fill in the elements using
        dagger of each estimated inner product. Thus the symmetry is maintained.
        In total, we are going to estimate
            total number of elements = 1/2 * (`tree_depth` + 1) * `tree_depth`
    """
    num_term = instance.get_num_term()
    coeffs = instance.get_coeffs()
    V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')
    for i in range(tree_depth):
        for j in range(i, tree_depth):
            element_counts = eval(counts[i][j], {"Counter": Counter})
            item = 0
            for k in range(num_term):
                for l in range(num_term):
                    ip_count = element_counts[k][l]
                    ip_r = __calculate_exp_by_count(ip_count[0])
                    ip_i = __calculate_exp_by_count(ip_count[1])
                    inner_product = ip_r + 1j * ip_i
                    item += conj(coeffs[k]) * coeffs[l] * inner_product
            V_dagger_V[i][j] = item
            if i < j:
                V_dagger_V[j][i] = conj(item)
    return V_dagger_V


def calculate_q_from_counts(instance, tree_depth, counts):
    num_term = instance.get_num_term()
    coeffs = instance.get_coeffs()
    q = zeros((tree_depth, 1), dtype='complex128')
    for i in range(tree_depth):
        element_counts = counts[i]
        item = 0
        for k in range(num_term):
            ip_id = eval(element_counts[k], {"Counter": Counter})
            ip_r = __calculate_exp_by_count(ip_id[0])
            ip_i = __calculate_exp_by_count(ip_id[1])
            inner_product = ip_r + 1j * ip_i
            item += conj(coeffs[k]) * inner_product
        q[i][0] = item
    return q


def find_true_loss_function(alphas):
    x = vstack((real(alphas), imag(alphas))).reshape(-1, 1)
    depth = len(alphas) - 1
    # Define the four sectors (quadrants)
    q1 = Q_noiseless[:depth + 1, :depth + 1]
    q2 = Q_noiseless[:depth + 1, 4:4 + depth + 1]
    q3 = Q_noiseless[4:4 + depth + 1, :depth + 1]
    q4 = Q_noiseless[4:4 + depth + 1, 4:4 + depth + 1]

    # Stack them back together
    top = hstack((q1, q2))
    bottom = hstack((q3, q4))
    Q_tem = vstack((top, bottom))

    r1 = r_noiseless[:depth + 1]
    r2 = r_noiseless[4:4 + depth + 1]

    r_tem = vstack((r1, r2)).reshape(-1, 1)
    xt = Tensor(x)
    Qt = Tensor(Q_tem) * 2
    rt = Tensor(r_tem) * (-2)
    return (0.5 * matmul(xt.T, matmul(Qt, xt)) + matmul(rt.T, xt) + 1).item()



def calculate_every_loss(Q, r):
    ALPHA = []
    LOSS = []
    LOSS_TRUE = []
    for depth in range(4):
        # Define the four sectors (quadrants)
        q1 = Q[:depth + 1, :depth + 1]
        q2 = Q[:depth + 1, 4:4 + depth + 1]
        q3 = Q[4:4 + depth + 1, :depth + 1]
        q4 = Q[4:4 + depth + 1, 4:4 + depth + 1]

        # Stack them back together
        top = hstack((q1, q2))
        bottom = hstack((q3, q4))
        Q_tem = vstack((top, bottom))

        r1 = r[:depth + 1]
        r2 = r[4:4 + depth + 1]

        r_tem = vstack((r1, r2))
        loss, alpha = solve_combination_parameters(Q_tem, r_tem, which_opt='ADAM')
        LOSS += [loss]
        ALPHA += [alpha]
        LOSS_TRUE += [find_true_loss_function(alpha)]
    return LOSS, LOSS_TRUE, ALPHA


with open('6_qubit_data_generation_matrix_A.csv', 'r', newline='') as csvfile:
    data_b = read_csv_b(6)
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    losses_noiseless=[]
    for i, row in enumerate(reader):
        if i == 2995:
            row_clean = [j for j in ''.join(row).split('"') if j != ',']
            nLc = row_clean[0].split(',')
            n = int(nLc[0])
            print("qubit number is:", n)
            L = int(nLc[1])
            print("term number is:", L)
            kappa = float(nLc[2])
            print('condition number is', kappa)
            pauli_strings = [__num_to_pauli_list(l) for l in eval(row_clean[1])]
            print('Pauli strings are:', pauli_strings)
            pauli_circuits = [__num_to_pauli_circuit(l) for l in eval(row_clean[1])]
            coeffs = [float(i) for i in eval(row_clean[2])]
            print('coefficients are:', coeffs)
            print()

            # circuit depth d
            d = 3
            ub = qasm3.loads(data_b.iloc[i].qasm)
            # print('Ub is given by:', data_b.iloc[i].b)

            # generate instance
            instance = Instance(n, L, kappa)
            instance.generate(given_coeffs=coeffs, given_unitaries=pauli_circuits, given_ub=ub)

            # retrieve hardware result
            V_dagger_V_counts_csv_filename = "V_dagger_V_counts.csv"
            q_counts_csv_filename = "q_counts.csv"
            V_dagger_V_counts = pd.read_csv(V_dagger_V_counts_csv_filename).values.tolist()
            q_counts = pd.read_csv(q_counts_csv_filename).values.tolist()


            V_dagger_V = calculate_V_dagger_V_from_counts(instance, 4, V_dagger_V_counts)
            q = calculate_q_from_counts(instance, 4, q_counts)
            Q, r = reshape_to_Q_r(V_dagger_V, q)
            # Create DataFrame
            Q_pd = pd.DataFrame(Q)
            r_pd = pd.DataFrame(r)
            # Save to CSV
            hardware_result_Q_csv_filename = "hardware_result_Q.csv"
            hardware_result_r_csv_filename = "hardware_result_r.csv"
            Q_pd.to_csv(hardware_result_Q_csv_filename, index=False)
            r_pd.to_csv(hardware_result_r_csv_filename, index=False)

            # calculate loss function for every depth
            LOSS, LOSS_TRUE, ALPHA = calculate_every_loss(Q, r)
            LOSS = pd.DataFrame(LOSS)
            LOSS_TRUE = pd.DataFrame(LOSS_TRUE)
            ALPHA = pd.DataFrame(ALPHA)

            # Save to CSV
            hardware_result_loss_csv_filename = "hardware_result_loss.csv"
            hardware_result_true_loss_csv_filename = "hardware_result_true_loss.csv"
            hardware_result_alpha_csv_filename = "hardware_result_alpha.csv"

            LOSS.to_csv(hardware_result_loss_csv_filename, index=False)
            LOSS_TRUE.to_csv(hardware_result_true_loss_csv_filename, index=False)
            ALPHA.to_csv(hardware_result_alpha_csv_filename, index=False)

            print("loss:", LOSS_TRUE)


