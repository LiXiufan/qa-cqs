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
    This is the calculator module to compute Q and r according to the Hadamard test shooting outcomes.
"""

from numpy import array
from numpy import zeros
from numpy import append

from numpy import real, imag
from numpy import conj

from hardware.execute import Hadamard_test
from braket.aws import AwsQuantumTask


def submit_all_inner_products_in_V_dagger_V(instance, ansatz_tree, **kwargs):
    r"""
        Estimate all independent inner products that appear in matrix `V_dagger_V`.
        Note that we only estimate all upper triangular elements and all diagonal elements.
        For lower elements, since the matrix `V_dagger_V` is symmetrical, we fill in the elements using
        dagger of each estimated inner product. Thus the symmetry is maintained.
        In total, we are going to estimate
            total number of elements = 1/2 * (`tree_depth` + 1) * `tree_depth`
    """
    unitaries = instance.get_unitaries()
    n = instance.get_num_qubit()
    num_term = instance.get_num_term()
    Ub = instance.get_ub()
    tree_depth = len(ansatz_tree)
    ip_idxes = [[0 for _ in range(tree_depth)] for _ in range(tree_depth)]
    for i in range(tree_depth):
        for j in range(i, tree_depth):
            element_idxes = [[0 for _ in range(num_term)] for _ in range(num_term)]
            for k in range(num_term):
                for l in range(num_term):
                    U1 = unitaries[k].compose(ansatz_tree[i], qubits=unitaries[k].qubits)
                    U2 = unitaries[l].compose(ansatz_tree[j], qubits=unitaries[l].qubits)
                    inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
                    element_idxes[k][l] = inner_product
            ip_idxes[i][j] = element_idxes
    return ip_idxes


def submit_all_inner_products_in_q(instance, ansatz_tree, **kwargs):
    r"""
        Estimate all independent inner products that appear in vector `q`
    """

    unitaries = instance.get_unitaries()
    n = instance.get_num_qubit()
    num_term = instance.get_num_term()
    Ub = instance.get_ub()
    tree_depth = len(ansatz_tree)
    ip_idxes = [0 for _ in range(tree_depth)]
    for i in range(tree_depth):
        element_idxes = [0 for _ in range(num_term)]
        for k in range(num_term):
            U1 = ansatz_tree[i]
            U2 = unitaries[k]
            inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
            element_idxes[k] = inner_product
        ip_idxes[i] = element_idxes
    return ip_idxes


def __retrieve_data(task):
    counter = task.result().measurement_counts
    prob_result = {str(outcome): int(counter[outcome]) for outcome in
                   counter.keys()}
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
        p0 = (p0 - 0.0048) / (1 - 2 * 0.0048)
        p1 = 1 - p0
    return p0 - p1


def retrieve_data(ip_id):
    task = AwsQuantumTask(arn=ip_id)
    status = task.state()
    if status != 'COMPLETED':
        return ValueError("I am sorry, your current task is in the status of", status)
    else:
        exp = __retrieve_data(task)
    return exp


def retrieve_and_estimate_V_dagger_V(instance, tree_depth, ip_idxes, backend='eigens'):
    r"""
        Estimate all independent inner products that appear in matrix `V_dagger_V`.
        Note that we only estimate all upper triangular elements and all diagonal elements.
        For lower elements, since the matrix `V_dagger_V` is symmetrical, we fill in the elements using
        dagger of each estimated inner product. Thus the symmetry is maintained.
        In total, we are going to estimate
            total number of elements = 1/2 * (`tree_depth` + 1) * `tree_depth`
    """
    if backend in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ip_idxes

    elif backend in ['aws-ionq-aria1']:
        num_term = instance.get_num_term()
        coeffs = instance.get_coeffs()
        V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')
        for i in range(tree_depth):
            for j in range(i, tree_depth):
                element_idxes = eval(ip_idxes[i][j])
                item = 0
                for k in range(num_term):
                    for l in range(num_term):
                        ip_id = element_idxes[k][l]
                        ip_r = retrieve_data(ip_id[0])
                        ip_i = retrieve_data(ip_id[1])
                        inner_product = ip_r + 1j * ip_i
                        item += conj(coeffs[k]) * coeffs[l] * inner_product
                V_dagger_V[i][j] = item
                if i < j:
                    V_dagger_V[j][i] = conj(item)
        return V_dagger_V
    else:
        return ValueError("Not supported backend.")


def retrieve_and_estimate_q(instance, tree_depth, ip_idxes, backend='eigens'):
    r"""
        Retrieve the results of all submitted tasks.
    """
    if backend in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ip_idxes

    elif backend in ['aws-ionq-aria1']:   # hardware retrieval
        num_term = instance.get_num_term()
        coeffs = instance.get_coeffs()
        q = zeros((tree_depth, 1), dtype='complex128')
        for i in range(tree_depth):
            element_idxes = ip_idxes[i]
            item = 0
            for k in range(num_term):
                ip_id = eval(element_idxes[k])
                ip_r = retrieve_data(ip_id[0])
                ip_i = retrieve_data(ip_id[1])
                inner_product = ip_r + 1j * ip_i
                item += conj(coeffs[k]) * inner_product
            q[i][0] = item
        return q
    else:
        return ValueError("Not supported backend.")



def reshape_to_Q_r(matrix, vector):
    Vr = real(matrix)
    Vi = imag(matrix)
    # Q = R  - I
    #     I   R
    Q = array(append(append(Vr, -Vi, axis=1), append(Vi, Vr, axis=1), axis=0), dtype='float64')

    qr = real(vector)
    qi = imag(vector)
    # r = Re(q)
    #     Im(q)
    r = array(append(qr, qi, axis=0), dtype='float64')
    return Q, r

