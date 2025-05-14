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


Q_noisy = array([[ 3.78307232e+01,  1.49353516e-02,  6.52541992e-01, -6.21453711e-01,
   7.39462109e-01,  5.88156055e-01,  2.39474396e+01, -7.59495312e-01],
 [ 1.49353516e-02,  3.82675193e+01,  6.43584570e-01, -2.37623242e-01,
  -5.88156055e-01,  1.31025508e+00,  1.89620684e+00,  2.18719043e+01],
 [ 6.52541992e-01,  6.43584570e-01,  3.69921545e+01, -1.83981953e+00,
  -2.39474396e+01, -1.89620684e+00,  1.23682383e+00, -1.59123984e+00],
 [-6.21453711e-01, -2.37623242e-01, -1.83981953e+00,  3.75601627e+01,
   7.59495312e-01, -2.18719043e+01,  1.59123984e+00, -8.71416211e-01],
 [-7.39462109e-01, -5.88156055e-01, -2.39474396e+01,  7.59495312e-01,
   3.78307232e+01,  1.49353516e-02,  6.52541992e-01, -6.21453711e-01],
 [ 5.88156055e-01, -1.31025508e+00, -1.89620684e+00, -2.18719043e+01,
   1.49353516e-02,  3.82675193e+01,  6.43584570e-01, -2.37623242e-01],
 [ 2.39474396e+01,  1.89620684e+00, -1.23682383e+00,  1.59123984e+00,
   6.52541992e-01,  6.43584570e-01,  3.69921545e+01, -1.83981953e+00],
 [-7.59495312e-01,  2.18719043e+01, -1.59123984e+00,  8.71416211e-01,
  -6.21453711e-01, -2.37623242e-01, -1.83981953e+00,  3.75601627e+01]])
r_noisy = array([[ 0.00931641],
 [ 3.53820312],
 [-0.34732422],
 [-0.08925781],
 [-0.09039063],
 [-0.20529297],
 [-0.11267578],
 [-3.12707031]])


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

            # the Ansatz tree depth is 4
            tree_depth = 4

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




