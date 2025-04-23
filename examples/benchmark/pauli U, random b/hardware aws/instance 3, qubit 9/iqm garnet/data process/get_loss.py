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


def recover_outcome(prob_result_pre, mea_corre):
    mea_corre_dict = {}
    q_count = 0
    for i in mea_corre:
        if i is not None:
            mea_corre_dict[str(i[0])] = int(i[1])
            q_count += 1

    prob_result = {}
    for outcome in prob_result_pre.keys():
        reduced_str_lst = [0 for _ in range(q_count)]
        value = prob_result_pre[outcome]
        string_list = list(outcome)
        for k in mea_corre_dict.keys():
            corre = mea_corre_dict[k]
            reduced_str_lst[corre] = string_list[int(k)]
        reduced_str = "".join(reduced_str_lst)
        if reduced_str in prob_result.keys():
            prob_result[reduced_str] += value
        else:
            prob_result[reduced_str] = value
    return prob_result


def __calculate_exp_by_count(count, mea_corre):
    prob_result_pre = {str(outcome): int(count[outcome]) for outcome in
                   count.keys()}
    prob_result = recover_outcome(prob_result_pre, mea_corre)
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
                    ip_mea_count = element_counts[k][l]
                    ip_count = ip_mea_count[0]
                    mea_corres = ip_mea_count[1]
                    ip_r = __calculate_exp_by_count(ip_count[0], mea_corres[0])
                    ip_i = __calculate_exp_by_count(ip_count[1], mea_corres[1])
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
            ip_mea_count = eval(element_counts[k], {"Counter": Counter})
            ip_count = ip_mea_count[0]
            mea_corres = ip_mea_count[1]
            ip_r = __calculate_exp_by_count(ip_count[0], mea_corres[0])
            ip_i = __calculate_exp_by_count(ip_count[1], mea_corres[1])
            inner_product = ip_r + 1j * ip_i
            item += conj(coeffs[k]) * inner_product
        q[i][0] = item
    return q


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
        loss, alpha = solve_combination_parameters(Q_tem, r_tem, which_opt='ADAM', reg=0.2)
        LOSS += [loss]
        ALPHA += [alpha]
        LOSS_TRUE += [find_true_loss_function(alpha, tree_depth)]
    return LOSS, LOSS_TRUE, ALPHA


with open('9_qubit_data_generation_matrix_A.csv', 'r', newline='') as csvfile:
    data_b = read_csv_b(9)
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    losses_noiseless=[]
    for i, row in enumerate(reader):
        if i == 2021:
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

            # the Ansatz tree depth is 10
            tree_depth = 10

            # retrieve hardware result
            V_dagger_V_counts_csv_filename = "V_dagger_V_counts.csv"
            q_counts_csv_filename = "q_counts.csv"
            V_dagger_V_counts = pd.read_csv(V_dagger_V_counts_csv_filename).values.tolist()
            q_counts = pd.read_csv(q_counts_csv_filename).values.tolist()

            V_dagger_V = calculate_V_dagger_V_from_counts(instance, tree_depth, V_dagger_V_counts)
            q = calculate_q_from_counts(instance, tree_depth, q_counts)
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
            LOSS, LOSS_TRUE, ALPHA = calculate_every_loss(Q, r, tree_depth)
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


