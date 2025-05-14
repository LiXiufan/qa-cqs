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
    This file is used for execution of noiseless simulation to solve different classes of linear systems equations
    on NUS HPC services.
"""

import csv
import qiskit.qasm3 as qasm3
import pathlib
import pandas as pd

from numpy import zeros, identity
from numpy import append
from numpy import conj
from tqdm import tqdm

from random import choice
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.random import random_circuit

from cvxopt import matrix
from cvxopt.solvers import qp
from numpy import linalg, diag, multiply, real, imag
from numpy import array, ndarray
from numpy import transpose, matmul
from sympy import Matrix
from typing import List, Tuple
import torch
import torch.optim as optim


class Instance:
    r"""Set the A matrix and unitary for b.

    This class generates the coefficient matrix A and unitary b of the linear system of equations
    according to the corresponding inputs.
    Users can also customize the A matrix with specific input.
    """

    def __init__(self, n, K, kappa):
        r"""Set the A matrix and unitary for b.

        This class generates the coefficient matrix A and unitary b of the linear system of equations
        according to the corresponding inputs.
        Users can also customize the A matrix with specific input.

        Args:
            n (int): qubit number
            K (int): number of decomposition terms
        """
        self.__which_type = None
        self.__unitaries = None  # unitaries
        self.__ub = None
        self.__matrix = None  # matrix
        self.__coeffs = None  # coefficients
        self.__num_term = K
        self.__num_qubit = n
        self.__cond_num = kappa
        self.__dim = 2 ** n

    def generate(self, given_coeffs=None, given_unitaries=None, given_ub=None):
        r"""Automatically generate a random matrix with the given intrinsic forms.

        Args:
            given_coeffs (List): a list of coefficients
            given_unitaries (List): a list of unitaries
            given_ub (List / QuantumCircuit): a unitary that corresponds to the state |b>
        """
        # Use given input coefficients and unitaries
        self.__coeffs = given_coeffs
        self.__unitaries = given_unitaries
        self.__ub = given_ub

    def get_unitaries(self):
        return self.__unitaries

    def get_coeffs(self):
        return self.__coeffs

    def get_num_qubit(self):
        return self.__num_qubit

    def get_num_term(self):
        return self.__num_term

    def get_ub(self):
        return self.__ub

    def get_cond_num(self):
        return self.__cond_num


def read_csv_b(n):
    # Get the working directory
    working_dir = pathlib.Path.cwd()

    # Move up the directory tree until reaching 'CQS_singapore'
    while working_dir.name != "qa-cqs" and working_dir != working_dir.parent:
        working_dir = working_dir.parent

    # Construct the new file path relative to 'CQS_singapore'
    file_path = working_dir / "instances_b" / (str(n)+"_b_random_circuits.csv")

    # Print the resolved path
    print(f"Resolved file path: {file_path}")

    # Check if the file exists
    if file_path.exists():
        print(f"Reading file: {file_path}")
        data_b = pd.read_csv(file_path)
        return data_b
    else:
        raise FileNotFoundError(f"Error: File not found at {file_path}")




# Hadamard test
def __build_circuit(n, U1, U2, Ub, alpha='r'):
    r"""
    Hadamard test to estimate <b| U1^{\dagger} U2 |b> given two unitaries U1, U2, and the state preparation circuit Ub.

    Args:
        n (int): qubit number
        U1 (QuantumCircuit): unitary of left vector that U1|0>=|v1>
        U2 (QuantumCircuit): unitary of right vector that U2|0>=|v2>
        U2 (QuantumCircuit): unitary of state preparation that Ub|0>=|b>
        alpha (str): 'r' or 'i', real or imaginary

    Returns:
        QuantumCircuit: the circuit of Hadamard test
    """
    if alpha not in ['r', 'i']:
        raise ValueError("Please specify the real part or the imaginary part using 'r' or 'i'.")
    anc = QuantumRegister(1, 'ancilla')
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(1, 'c')
    cir = QuantumCircuit(anc, qr, cr)
    cir.h(anc[0])
    cir.append(Ub.to_gate(), [*qr])
    cir.append(U1.to_gate().control(ctrl_state='0'), [anc[0], *qr])
    cir.append(U2.to_gate().control(ctrl_state='1'), [anc[0], *qr])
    if alpha == 'i':
        cir.sdg(anc[0])
    cir.h(anc[0])
    cir.measure(anc[0], cr[0])
    return cir

def __run_circuit(cir, shots=1024):
    # Transpile for simulator
    simulator = AerSimulator()
    cir = transpile(cir, simulator)
    if shots == 0:
        # Run and get probabilities
        cir.remove_final_measurements()
        state_vec = Statevector(cir)
        prob_zero_qubit = state_vec.probabilities([0])
        p0 = prob_zero_qubit[0]
        p1 = prob_zero_qubit[1]
    else:
        # Run and get counts
        result = simulator.run(cir, shots=shots).result()
        counts = result.get_counts(0)
        if '0' not in counts.keys():
            p0 = 0
            p1 = 1
        elif '1' not in counts.keys():
            p0 = 1
            p1 = 0
        else:
            p0 = counts['0'] / shots
            p1 = counts['1'] / shots
    return p0 - p1

def Hadamard_test_qiskit_noiseless(n, U1, U2, Ub, shots=1024):
    cir_r = __build_circuit(n, U1, U2, Ub, alpha='r')
    exp_r = __run_circuit(cir_r, shots=int(shots / 2))
    cir_i = __build_circuit(n, U1, U2, Ub, alpha='i')
    exp_i = __run_circuit(cir_i, shots=int(shots / 2))
    expec = exp_r + exp_i * 1j
    return expec

def Hadamard_test(n, U1, U2, Ub, real=None, backend=None, shots=None, **kwargs):
    if real is None:
        real = True
    # select the backend
    if backend == 'qiskit-noiseless':
        return Hadamard_test_qiskit_noiseless(n, U1, U2, Ub, shots)

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



def U_list_dagger(U):
    return U[::-1]


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

    # Calculate total number of iterations
    total_iterations = sum((tree_depth - i) * (num_term ** 2) for i in range(tree_depth))

    # Initialize tqdm progress bar
    with tqdm(total=total_iterations, desc="Hadamard tests V_dagger_V Progress") as pbar:
        for i in range(tree_depth):
            for j in range(i, tree_depth):
                element_idxes = [[0 for _ in range(num_term)] for _ in range(num_term)]
                for k in range(num_term):
                    for l in range(num_term):
                        U1 = unitaries[k].compose(ansatz_tree[i], qubits=unitaries[k].qubits)
                        U2 = unitaries[l].compose(ansatz_tree[j], qubits=unitaries[l].qubits)
                        inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
                        element_idxes[k][l] = inner_product
                        pbar.update(1)  # Update global progress bar
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
    # Initialize tqdm progress bar
    with tqdm(total=tree_depth*num_term, desc="Hadamard tests q Progress") as pbar:
        for i in range(tree_depth):
            element_idxes = [0 for _ in range(num_term)]
            for k in range(num_term):
                U1 = ansatz_tree[i]
                U2 = unitaries[k]
                inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
                element_idxes[k] = inner_product
                pbar.update(1)  # Update global progress bar
            ip_idxes[i] = element_idxes
    return ip_idxes


def __retrieve__all_inner_products_in_V_dagger_V(instance, ansatz_tree, ip_idxes, backend='eigens'):
    r"""
        Retrieve the results of all submitted tasks.
    """
    if backend in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ip_idxes
    else:  # hardware retrieval
        return 0


def __retrieve__all_inner_products_in_q(instance, ansatz_tree, ip_idxes, backend='eigens'):
    r"""
        Retrieve the results of all submitted tasks.
    """
    if backend in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ip_idxes
    else:  # hardware retrieval
        return 0


def __estimate_V_dagger_V(instance, ansatz_tree, loss_type=None, backend=None, **kwargs):
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
    tree_depth = len(ansatz_tree)
    V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')

    ip_idxes = submit_all_inner_products_in_V_dagger_V(instance, ansatz_tree, backend=backend, **kwargs)
    ip_values = __retrieve__all_inner_products_in_V_dagger_V(instance, ansatz_tree, ip_idxes, backend=backend)

    for i in range(tree_depth):
        for j in range(i, tree_depth):
            element_values = ip_values[i][j]
            item = 0
            for k in range(num_term):
                for l in range(num_term):
                    inner_product = element_values[k][l]
                    item += conj(coeffs[k]) * coeffs[l] * inner_product
            V_dagger_V[i][j] = item
            if i < j:
                V_dagger_V[j][i] = conj(item)
    if loss_type == 'l2reg':
        V_dagger_V += 1 / 2 * identity(tree_depth, dtype='complex128')
    return V_dagger_V


def __estimate_q(instance, ansatz_tree, backend=None, **kwargs):
    r"""
        Estimate all independent inner products that appear in vector `q`.
        In total, we are going to estimate
            total number of elements = `tree_depth`
    """

    num_term = instance.get_num_term()
    coeffs = instance.get_coeffs()
    tree_depth = len(ansatz_tree)
    q = zeros((tree_depth, 1), dtype='complex128')

    ip_idxes = submit_all_inner_products_in_q(instance, ansatz_tree, backend=backend, **kwargs)
    ip_values = __retrieve__all_inner_products_in_q(instance, ansatz_tree, ip_idxes, backend=backend)


    for i in range(tree_depth):
        element_values = ip_values[i]
        item = 0
        for k in range(num_term):
            inner_product = element_values[k]
            item += conj(coeffs[k]) * inner_product
        q[i][0] = item
    return q


def __reshape_to_Q_r(matrix, vector):
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


def calculate_Q_r(instance, ansatz_tree, loss_type=None, **kwargs):
    r"""
        Please note that the objective function of CVXOPT has the form:   1/2  x^T P x  +   q^T x
        But our objective function is:                                         z^T Q z  - 2 r^T z + 1
        So the coefficients are corrected here.

    :param R:
    :param I:
    :param q:
    :return: Q, r
    """
    V_dagger_V = __estimate_V_dagger_V(instance, ansatz_tree, loss_type=loss_type, **kwargs)
    q = __estimate_q(instance, ansatz_tree, **kwargs)
    Q, r = __reshape_to_Q_r(V_dagger_V, q)
    return Q, r




def solve_combination_parameters(Q: ndarray, r: ndarray, which_opt=None) -> Tuple[float, List]:
    r"""Optimization module for solving the optimal combination parameters.

    In this module, we implement the CVXOPT package as an external resource package.
    CVXOPT is a free software package for convex optimization based on the Python programming language.
    Reference: https://cvxopt.org
    MIT Course: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

    CVXOPT Notation of a quadratic optimization problem:
                min    1/2  x^T Q x + q^T r
          subject to   Gx  <=  h
                       Ax  =  b

    Args:
        Q (np.ndarray): the auxiliary matrix Q
        r (np.ndarray): the auxiliary vector r

    Returns:
        Tuple[float, List]: loss and the optimal combination parameters
    """
    if which_opt is None:
        which_opt = 'cvxopt'

    if which_opt == 'cvxopt':
        Q = 2 * matrix(Q)
        r = (-2) * matrix(r)
        # Solve the optimization problem using the kkt solver with regularization constant of 1e-12
        # Note: for more realistic experiments, due to the erroneous results,
        # it is suggested to change the regularization constant to get a better performance.
        # comb_params = qp(Q, r, kktsolver='ldl', options={'kktreg': 1e-16})['x']
        comb_params = qp(Q, r, kktsolver='ldl', options={'kktreg': 1e-15})['x']



    elif which_opt == "ADAM":

        Q = 2 * torch.Tensor(Q)

        r = (-2) * torch.Tensor(r)

        # Define the loss function L(x) = x^T Q x + r^T x

        def loss_function(x, Q, r):

            return torch.abs(0.5 * torch.matmul(x.T, torch.matmul(Q, x)) + torch.matmul(r.T, x) + 1)

        # Initialize x, Q, and r

        dim = len(r)  # Dimension of x

        comb_params = torch.randn(dim, requires_grad=True)  # Trainable variable

        # Define optimizer

        optimizer = optim.Adam([comb_params], lr=0.02)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)  # Reduce LR every 1000 epochs

        max_epochs = 10 ** 6

        patience_epochs = 1500  # Check progress every 1500 epochs

        threshold = 0.95  # Loss must be at most 95% of the previous one

        # Training loop

        best_loss = float("inf")  # Keep track of the best loss

        last_improvement_epoch = 0  # Track last improvement

        for epoch in range(max_epochs):

            optimizer.zero_grad()

            loss = loss_function(comb_params, Q, r)  # Compute loss

            loss.backward()  # Compute gradients

            optimizer.step()  # Update x

            scheduler.step()  # Adjust learning rate

            # Check stopping criterion

            if loss.item() < best_loss * threshold:
                best_loss = loss.item()

                last_improvement_epoch = epoch  # Reset improvement tracker

            if epoch - last_improvement_epoch >= patience_epochs:
                # print(f"Stopping early at epoch {epoch + 1}, loss stagnated at {loss.item():.6f}")

                break
        print(best_loss)
        comb_params = comb_params.tolist()
        # Train x
    elif which_opt == 'inv':
        Q = Matrix(Q)
        P, D = Q.diagonalize()
        D = array(D, dtype='complex128')
        D_diag = diag(D)
        D_diag_inv = []
        for d in D_diag:
            if linalg.norm(d) <= 1e-12:
                d_inv = 0
            else:
                if linalg.norm(real(d)) <= 1e-12:
                    d = 0 + imag(d) * 1j
                if linalg.norm(imag(d)) <= 1e-12:
                    d = real(d)
                d_inv = 1 / d
            D_diag_inv.append(d_inv)
        D_diag_inv = array(D_diag_inv, dtype='complex128')
        comb_params = multiply(D_diag_inv, r.reshape(-1))
    else:
        raise ValueError

    half_var = int(len(comb_params) / 2)
    alphas = [0 for _ in range(half_var)]

    for i in range(half_var):
        alpha = comb_params[i] + comb_params[half_var + i] * 1j
        alphas[i] = alpha

    # Calculate l2-norm loss function
    params_array = array(comb_params).reshape(-1, 1)
    Q_array = array(Q)
    r_array = array(r).reshape(-1, 1)
    loss = abs((matmul(matmul(transpose(params_array), 0.5*Q_array), params_array)
                + matmul(transpose(r_array), params_array) + 1).item())
    # Calculate Hamiltonian loss fucntion
    # loss = abs((transpose(params_array) @ Q_array @ params_array - (transpose(params_array) @ r_array) * transpose(r_array) @ params_array).item())
    return loss, alphas


def __number_to_base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def __get_idx(num_term, tree_depth):
    current_depth = 0
    while True:
        current_depth += 1
        l_ = (1 - num_term ** (current_depth - 1)) / (1 - num_term)
        h_ = (1 - num_term ** current_depth) / (1 - num_term)
        if l_ < tree_depth <= h_:
            # print("depth is:", current_depth)
            idx = tree_depth - l_ - 1
            digits = __number_to_base(idx, num_term)
            digits = [0 for _ in range(current_depth - 1 - len(digits))] + digits
            return digits
        else:
            continue


def __expand_breadth_first(instance, ansatz_tree):
    tree_depth = len(ansatz_tree)
    num_term = instance.get_num_term()
    unitaries = instance.get_unitaries()
    next_depth = tree_depth + 1
    idx = __get_idx(num_term, next_depth)
    child_node = ansatz_tree[0][:]
    for i in idx:
        child_node += unitaries[i]
    ansatz_tree.append(child_node)
    return ansatz_tree


def __obtain_child_space(instance, ansatz_tree):
    parent_node = ansatz_tree[-1]
    num_term = instance.get_num_term()
    unitaries = instance.get_unitaries()
    child_space = [parent_node.compose(unitaries[i], qubits=parent_node.qubits) for i in range(num_term)]
    return child_space


def __calculate_grad_overlaps(instance, ansatz_tree, **kwargs):
    r"""
        Estimate all independent inner products that appear in term 1 and term 2.
        In total, we are going to estimate
            total number of elements = num_term * (tree_depth * num_term * num_term + num_term)
    """

    unitaries = instance.get_unitaries()
    n = instance.get_num_qubit()
    num_term = instance.get_num_term()
    Ub = instance.get_ub()
    tree_depth = len(ansatz_tree)
    parent_node = ansatz_tree[-1]
    child_space = [parent_node.compose(unitaries[i], qubits=parent_node.qubits) for i in range(num_term)]

    # construct the structure of the list to record the indexes
    grad_overlaps = [
        [
            [
                [
                    [
                        0 for _ in range(num_term)
                    ]
                    for _ in range(num_term)
                ]
                for _ in range(tree_depth)
            ],
            [
                0 for _ in range(num_term)
            ]
        ] for _ in range(num_term)]

    for i in range(num_term):
        term_idxes = grad_overlaps[i]
        term_1_idxes = term_idxes[0]
        term_2_idxes = term_idxes[1]
        U1 = child_space[i]
        for j in range(tree_depth):
            for k in range(num_term):
                for l in range(num_term):
                    U2 = unitaries[k].compose(unitaries[l], qubits=unitaries[k].qubits).compose(ansatz_tree[j], qubits=unitaries[k].qubits)
                    inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
                    term_1_idxes[j][k][l] = inner_product
        for j in range(num_term):
            U2 = unitaries[j]
            inner_product = Hadamard_test(n, U1, U2, Ub, **kwargs)
            term_2_idxes[j] = inner_product
    return grad_overlaps


def calculate_gradient_overlaps(instance, alphas, ansatz_tree, backend=None, **kwargs):
    r"""
        We would like to estimate the gradient overlap between each of the child nodes with respect to the parent node.
            grad_overlap = 2 <child| A A x - 2 <child| A b
    """
    # submit the quantum tasks
    grad_overlaps = __calculate_grad_overlaps(instance, ansatz_tree, backend=backend, **kwargs)
    # calculate gradient overlap
    coeffs = instance.get_coeffs()
    num_term = instance.get_num_term()
    tree_depth = len(ansatz_tree)
    gradient_overlaps = [0 for _ in range(num_term)]
    for i in range(num_term):
        term_values = grad_overlaps[i]
        term_1_values = term_values[0]
        term_2_values = term_values[1]
        term_1 = 0
        for j in range(tree_depth):
            term_1_1 = 0
            for k in range(num_term):
                for l in range(num_term):
                    inner_product = term_1_values[j][k][l]
                    term_1_1 += conj(coeffs[k]) * coeffs[l] * inner_product
            term_1 += alphas[j] * term_1_1
        term_2 = 0
        for j in range(num_term):
            inner_product = term_2_values[j]
            term_2 += coeffs[j] * inner_product
        gradient_overlap = abs(2 * term_1 - 2 * term_2)
        gradient_overlaps[i] = gradient_overlap
    return gradient_overlaps


def __expand_by_gradient(instance, alphas, ansatz_tree, **kwargs):
    # construct child space
    child_space = __obtain_child_space(instance, ansatz_tree)
    # calculate gradient overlaps
    gradient_overlaps = calculate_gradient_overlaps(instance, alphas, ansatz_tree, **kwargs)
    # To consider the case when there are several candidates for the child node.
    max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]
    idx = choice(max_index)
    child_node_opt = child_space[idx]
    ansatz_tree.append(child_node_opt)
    return ansatz_tree


def expand_ansatz_tree(instance, alphas, ansatz_tree, mtd=None, **kwargs):
    if mtd is None:
        mtd = 'gradient'
    if mtd == 'breadth':
        ansatz_tree = __expand_breadth_first(instance, ansatz_tree)
    elif mtd == 'gradient':
        ansatz_tree = __expand_by_gradient(instance, alphas, ansatz_tree, **kwargs)
    else:
        return ValueError(
            "Please specify the Ansatz tree expansion method by setting `mtd` to be in ['breadth', 'gradient'].")
    return ansatz_tree

def main_solver(instance, ansatz_tree, **kwargs):
    r"""
        This function solves Ax=b when Ansatz tree is known to us.
    """
    # Performing Hadamard test to calculate Q and r
    Q, r = calculate_Q_r(instance, ansatz_tree, **kwargs)
    # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
    loss, alphas = solve_combination_parameters(Q, r, which_opt='ADAM')
    print("loss:", loss)
    print("combination parameters are:", alphas)
    return loss, alphas


def __solve_and_expand(instance, ansatz_tree, **kwargs):
    loss, alphas = main_solver(instance, ansatz_tree, **kwargs)
    new_ansatz_tree = expand_ansatz_tree(instance, alphas, ansatz_tree, **kwargs)
    return loss, alphas, new_ansatz_tree


def main_prober(instance, backend=None, ITR=None, eps=None, **kwargs):
    r"""
        This function solves Ax=b when Ansatz tree is not known to us.
        Thus we use expansion algorithms to probe the solution space,
        including breadth-first search, gradient heuristic, etc.
    """
    n = instance.get_num_qubit()
    # For eigens simulator, each unitary gate is a list of Pauli strings.
    # For qiskit simulator, each unitary gate is a quantum circuit object in qiskit.
    if backend not in ['eigens', 'qiskit-noiseless', 'qiskit-noisy']:
        return ValueError("We do not allow the calls of this function in real ionq aria, since the "
                          "connection and execution takes time. Instead, we encourage the user to separate `submit`"
                          "process and `retrieve` process. Please try to build it using `main-solver`.")
    if eps is None:
        eps = 0.01

    # ansatz tree only contains identity at the beginning
    if backend == 'eigens':
        ansatz_tree = [[["I" for _ in range(n)]]]
    else:
        qr = QuantumRegister(n, 'q')
        id_cir = QuantumCircuit(qr)
        id_cir.id(qr)
        ansatz_tree = [id_cir]

    # main procedure
    LOSS = []
    Itr = []
    itr_count = 0
    loss = 1
    alphas = 0
    if ITR is None:
        while loss > eps:
            itr_count += 1
            Itr.append(itr_count)
            loss, alphas, ansatz_tree = __solve_and_expand(instance, ansatz_tree, backend=backend, **kwargs)
            # print("loss:", loss)
            LOSS.append(loss)
    else:
        for itr in range(1, ITR + 1):
            Itr.append(itr)
            loss, alphas, ansatz_tree = __solve_and_expand(instance, ansatz_tree, backend=backend, **kwargs)
            # print("loss:", loss)
            LOSS.append(loss)
            if loss < eps:
                break
    # print("combination parameters are:", alphas)

    return Itr, LOSS, ansatz_tree


with open('3_qubit_data_generation_matrix_A.csv', 'r', newline='') as csvfile:
    data_b=read_csv_b(3)
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i, row in enumerate(reader):
        if 2 > i > 0:
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
            ub = qasm3.loads(data_b.iloc[i].qasm)#random_circuit(num_qubits=3, max_operands=2, depth=3, measure=False)
            print('Ub is given by:', data_b.iloc[i].b)
            print(ub)
            # generate instance
            instance = Instance(n, L, kappa)
            instance.generate(given_coeffs=coeffs, given_unitaries=pauli_circuits, given_ub=ub)
            Itr, LOSS, ansatz_tree = main_prober(instance, backend='qiskit-noiseless', ITR=20, shots=0, optimization_level=2)
            print(Itr)
            print(LOSS)
            # matrix = instance.get_matrix()
            # print("The first example returns with a matrix:")
            # print(matrix)
            # print()














