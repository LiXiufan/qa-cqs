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
    Expansion of the solution space.
    Here we would like to introduce two types of expansion methods.
    One is originated from the article: Ansatz tree with maximum gradient overlap
    Another is our self-developed method: Ansatz tree with l1 sampling expansion.
"""
from numpy import conj
# from cqs.calculation import U_list_dagger, calculate_statistics
from random import choice
from hardware.execute import Hadamard_test


#
# from numpy import random, linalg, sqrt, kron
# from numpy import real, conj, transpose
# from utils import draw_ansatz_tree
# from qiskit_ionq import IonQProvider
# from qiskit_braket_provider import AWSBraketProvider

# unitaries = [['X', 'Z'], ['Z', 'X'], ['I', 'Y']]
# alphas = [1.6317321653264614, 0.2249104575899552, 1.7897631234464821]

# BRAKET_DEVICE = 'SV1'
# BRAKET_DEVICE = 'Aria 1'
# BRAKET_DEVICE = 'Harmony'

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

def __calculate_gradient_overlaps(instance, alphas, ansatz_tree):
    coeffs = instance.get_coeffs()
    unitaries = instance.get_unitaries()
    n = instance.get_num_qubit()
    num_term = instance.get_num_term()
    tree_depth = len(ansatz_tree)
    parent_node = ansatz_tree[-1]
    child_space = [parent_node + unitaries[i] for i in range(num_term)]
    gradient_overlaps = [0 for _ in range(num_term)]
    for i in range(num_term):
        term_1 = 0
        for j in range(tree_depth):
            term_1_1 = 0
            for k in range(num_term):
                for l in range(num_term):
                    U1 = child_space[i]
                    U2 = unitaries[k] + unitaries[l] + ansatz_tree[j]
                    inner_product = Hadamard_test(n, U1, U2, real=None, backend='eigens', shots=None, device=None)
                    term_1_1 += conj(coeffs[k]) * coeffs[l] * inner_product
            term_1 += alphas[j] * term_1_1
        term_2 = 0
        for j in range(num_term):
            U1 = child_space[i]
            U2 = unitaries[j]
            inner_product = Hadamard_test(n, U1, U2, real=None, backend='eigens', shots=None, device=None)
            term_2 += coeffs[j] * inner_product
        gradient_overlap = abs(2 * term_1 - 2 * term_2)
        gradient_overlaps[i] = gradient_overlap
    return child_space, gradient_overlaps

def __expand_by_gradient(instance, alphas, ansatz_tree):
    # Create child space and calculate gradient overlaps
    child_space, gradient_overlaps = __calculate_gradient_overlaps(instance, alphas, ansatz_tree)
    # To consider the case when there are several candidates for the child node.
    max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]
    idx = choice(max_index)
    child_node_opt = child_space[idx]
    ansatz_tree.append(child_node_opt)
    return ansatz_tree

def expand_ansatz_tree_by_eigens(instance, alphas, ansatz_tree, mtd=None):
    if mtd is None:
        mtd = 'gradient'

    if mtd == 'breadth':
        return __expand_breadth_first(instance, ansatz_tree)
    elif mtd == 'gradient':
        return __expand_by_gradient(instance, alphas, ansatz_tree)
























#
# def expand_ansatz_tree(A, vars, ansatz_tree, loss_type=None, backend=None, draw_tree=False, shots_budget=1024,
#                        frugal=False, mtd=None, tasks_num=0, shots_num=0, file_name='message.txt'):
#     if backend is None:
#         backend = 'eigens'
#     if mtd is None:
#         mtd = 'gradient'
#
#     A_coeffs = A.get_coeff()
#     A_unitaries = A.get_unitary()
#     A_terms_number = len(A_coeffs)
#     tree_depth = len(ansatz_tree)
#     if draw_tree is True:
#         draw_ansatz_tree(A_terms_number, tree_depth, 'Expansion')
#     if mtd == 'gradient':
#         parent_node = ansatz_tree[-1]
#         child_space = [parent_node + A_unitaries[i] for i in range(A_terms_number)]
#
#         if frugal is True:
#             shots_each_entry = shots_budget / (10 * len(child_space))
#
#             M_Child_Nodes = sum([abs(A_coeffs[k] * A_coeffs[l] * vars[j])
#                                  for j in range(tree_depth)
#                                  for k in range(A_terms_number)
#                                  for l in range(A_terms_number)] +
#                                 [abs(A_coeffs[j]) for j in range(A_terms_number)])
#
#             P_Child_Nodes_Term_1 = [
#                 10 * int(shots_each_entry * abs(vars[j] * A_coeffs[k] * A_coeffs[l]) / M_Child_Nodes)
#                 for j in range(tree_depth)
#                 for k in range(A_terms_number)
#                 for l in range(A_terms_number)]
#             P_Child_Nodes_Term_2 = [10 * int(shots_each_entry * abs(A_coeffs[j]) / M_Child_Nodes) for j in
#                                     range(A_terms_number)]
#
#         else:
#             # Uniform distribution
#             shots_ave = 100
#             P_Child_Nodes_Term_1 = [shots_ave
#                                     for _ in range(tree_depth)
#                                     for _ in range(A_terms_number)
#                                     for _ in range(A_terms_number)]
#             P_Child_Nodes_Term_2 = [shots_ave for j in range(A_terms_number)]
#
#         gradient_overlaps = [0 for _ in range(len(child_space))]
#
#         for i, child_node in enumerate(child_space):
#             if backend == 'ionq' or backend == 'braket':
#                 Job_ids_1_R = []
#                 Job_ids_1_I = []
#                 Job_ids_2_R = []
#                 Job_ids_2_I = []
#                 Job_ids_reg_R = []
#                 Job_ids_reg_I = []
#
#                 for j in range(tree_depth):
#                     anstaz_state = ansatz_tree[j]
#                     shots = 100
#                     if loss_type == 'l2reg':
#                         u = U_list_dagger(child_node) + anstaz_state
#                         jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                       tasks_num=tasks_num, shots_num=shots_num)
#                         Job_ids_reg_R.append(jobid_R)
#                         jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                       tasks_num=tasks_num, shots_num=shots_num)
#                         Job_ids_reg_I.append(jobid_I)
#
#                     for k in range(A_terms_number):
#                         for l in range(A_terms_number):
#                             shots = P_Child_Nodes_Term_1[j * A_terms_number * A_terms_number + k * A_terms_number + l]
#                             u = U_list_dagger(child_node) + A_unitaries[k] + A_unitaries[l] + anstaz_state
#                             # file1 = open(file_name, "a")
#                             # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
#                             # file1.close()
#                             jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                           tasks_num=tasks_num, shots_num=shots_num)
#                             Job_ids_1_R.append(jobid_R)
#                             jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                           tasks_num=tasks_num, shots_num=shots_num)
#                             Job_ids_1_I.append(jobid_I)
#
#                 for j in range(A_terms_number):
#                     shots = P_Child_Nodes_Term_2[j]
#                     u = U_list_dagger(child_node) + A_unitaries[j]
#                     shots = 100
#                     # file1 = open(file_name, "a")
#                     # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
#                     # file1.close()
#                     jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     Job_ids_2_R.append(jobid_R)
#                     jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                   tasks_num=tasks_num, shots_num=shots_num)
#                     Job_ids_2_I.append(jobid_I)
#
#                 if backend == 'ionq':
#                     provider = IonQProvider('pUhwyKCHRYAvWUChFqwTApQwow4mS2h7')
#                     # simulator_backend = provider.get_backend("ionq_qpu.harmony")
#                     # simulator_backend = provider.get_backend("ionq_qpu.aria-1")
#                     simulator_backend = provider.get_backend("ionq_simulator")
#                 else:
#                     provider = AWSBraketProvider()
#                     simulator_backend = provider.get_backend(BRAKET_DEVICE)
#
#                 exp_1_R = calculate_statistics(simulator_backend, Job_ids_1_R)
#                 exp_1_I = calculate_statistics(simulator_backend, Job_ids_1_I)
#                 exp_2_R = calculate_statistics(simulator_backend, Job_ids_2_R)
#                 exp_2_I = calculate_statistics(simulator_backend, Job_ids_2_I)
#                 exp_reg_R = calculate_statistics(simulator_backend, Job_ids_reg_R)
#                 exp_reg_I = calculate_statistics(simulator_backend, Job_ids_reg_I)
#
#                 term_1 = 0
#                 for j in range(tree_depth):
#                     term_1_1 = 0
#                     alpha = vars[j]
#                     for k in range(A_terms_number):
#                         for l in range(A_terms_number):
#                             beta_k = A_coeffs[k]
#                             beta_l = A_coeffs[l]
#                             inner_product_real = exp_1_R[j * A_terms_number * A_terms_number + k * A_terms_number + l]
#                             inner_product_imag = exp_1_I[j * A_terms_number * A_terms_number + k * A_terms_number + l]
#                             inner_product = inner_product_real - inner_product_imag * 1j
#                             term_1_1 += beta_k * beta_l * inner_product
#                     term_1 += alpha * term_1_1
#
#                 term_2 = 0
#                 for j in range(A_terms_number):
#                     beta_j = A_coeffs[j]
#                     inner_product_real = exp_2_R[j]
#                     inner_product_imag = exp_2_I[j]
#                     inner_product = inner_product_real - inner_product_imag * 1j
#                     term_2 += beta_j * inner_product
#
#                 if loss_type == 'l2reg':
#                     term_reg = 0
#                     for j in range(tree_depth):
#                         alpha = vars[j]
#                         inner_product_real = exp_reg_R[j]
#                         inner_product_imag = exp_reg_I[j]
#                         inner_product = inner_product_real - inner_product_imag * 1j
#                         term_reg += alpha * inner_product
#
#             else:
#                 term_1 = 0
#                 term_reg = 0
#                 for j in range(tree_depth):
#                     term_1_1 = 0
#                     alpha = vars[j]
#                     anstaz_state = ansatz_tree[j]
#
#                     if loss_type == 'l2reg':
#                         u = U_list_dagger(child_node) + anstaz_state
#                         shots = 100
#                         inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1,
#                                                                                  shots=shots,
#                                                                                  tasks_num=tasks_num,
#                                                                                  shots_num=shots_num)
#                         inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j,
#                                                                                  shots=shots,
#                                                                                  tasks_num=tasks_num,
#                                                                                  shots_num=shots_num)
#                         inner_product = inner_product_real - inner_product_imag * 1j
#                         term_reg += alpha * inner_product
#
#                     for k in range(A_terms_number):
#                         for l in range(A_terms_number):
#                             beta_k = A_coeffs[k]
#                             beta_l = A_coeffs[l]
#                             shots = P_Child_Nodes_Term_1[j * A_terms_number * A_terms_number + k * A_terms_number + l]
#                             shots = 100
#                             u = U_list_dagger(child_node) + A_unitaries[k] + A_unitaries[l] + anstaz_state
#                             # file1 = open(file_name, "a")
#                             # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
#                             # file1.close()
#                             inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1,
#                                                                                      shots=shots, tasks_num=tasks_num,
#                                                                                      shots_num=shots_num)
#                             inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j,
#                                                                                      shots=shots, tasks_num=tasks_num,
#                                                                                      shots_num=shots_num)
#                             inner_product = inner_product_real - inner_product_imag * 1j
#                             term_1_1 += beta_k * beta_l * inner_product
#                     term_1 += alpha * term_1_1
#
#                 term_2 = 0
#                 for j in range(A_terms_number):
#                     beta_j = A_coeffs[j]
#                     shots = P_Child_Nodes_Term_2[j]
#                     shots = 100
#                     u = U_list_dagger(child_node) + A_unitaries[j]
#                     # file1 = open(file_name, "a")
#                     # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
#                     # file1.close()
#                     inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots,
#                                                                              tasks_num=tasks_num, shots_num=shots_num)
#                     inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots,
#                                                                              tasks_num=tasks_num, shots_num=shots_num)
#                     inner_product = inner_product_real - inner_product_imag * 1j
#                     term_2 += beta_j * inner_product
#
#             gradient_overlap = abs(2 * term_1 - 2 * term_2 + term_reg)
#             gradient_overlaps[i] = gradient_overlap
#
#         # print("")
#         # print("Child space is:", child_space)
#         # print("")
#         # print("Gradient Overlaps are:", gradient_overlaps)
#         # print("")
#         # To consider the case when there are several candidates for the child node.
#         max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]
#         idx = random.choice(max_index)
#         child_node_opt = child_space[idx]
#
#         # print("Choose the child node of maximum overlap:", child_node_opt)
#         # print("")
#         file1 = open(file_name, "a")
#         file1.writelines(["Child space is:", str(child_space), '\n'])
#         file1.writelines(["Gradient Overlaps are:", str(gradient_overlaps), '\n'])
#         file1.writelines(["Choose the child node of maximum overlap:", str(child_node_opt), '\n\n'])
#         file1.close()
#
#         if draw_tree is True:
#             draw_ansatz_tree(A_terms_number, tree_depth, 'Optimal', which_index=idx)
#         ansatz_tree.append(child_node_opt)
#         if draw_tree is True:
#             draw_ansatz_tree(A_terms_number, tree_depth, 'Pending')
#
#     elif mtd == 'breadth':
#         next_depth = tree_depth + 1
#         idx = get_idx(A_terms_number, next_depth)
#         child_node = ansatz_tree[0][:]
#         for i in idx:
#             child_node += A_unitaries[i]
#         ansatz_tree.append(child_node)
#     return ansatz_tree, tasks_num, shots_num

# def optimize_with_stochastic_descend(A, N):
#     # Use L1 sampling to do the expansion.
#     # c.f. https://arxiv.org/abs/1907.05378
#     A_coeffs = A.get_coeff()
#     A_unitaries = A.get_unitary()
#
#     width = len(A_unitaries[0][0])
#     zeros = zero_state()
#     if width > 1:
#         for j in range(width - 1):
#             zeros = kron(zeros, zero_state())
#
#     # xt = array([0 for _ in range(2 ** width)]).reshape(-1, 1)
#     xt = zeros
#     XT = [xt]
#
#     # print("B is", B)
#     # print('eta is:', eta)
#     # print('Upper bound is:', error)
#
#     A_coeffs_norm = [coeff / sum(A_coeffs) for coeff in A_coeffs]
#     unitary_indexes = list(range(len(A_unitaries)))
#
#     A_mat = A.get_matrix()
#
#     # In order to quantify the L2, we will have to pre-know the optimal x by matrix calculations
#     x_opt = linalg.inv(A_mat) @ zeros
#
#     for t in range(1, N):
#         L2 = linalg.norm(xt - x_opt)
#         B = 4 * (L2 ** 2) + 4
#         eta = L2 / (B * sqrt(N))
#         # print(L2)
#         U1, U2, U3 = [A_unitaries[random.choice(unitary_indexes, p=A_coeffs_norm)] for _ in range(3)]
#         gt = 2 * get_unitary(U1) @ get_unitary(U2) @ xt - 2 * get_unitary(U3) @ zeros
#         xt = xt - eta * gt
#         # print('eta is:', eta)
#         XT.append(xt)
#
#     # print(XT)
#     # print()
#     # print(sum(XT))
#     # print()
#     xt_ave = sum(XT) / N
#     # print(xt_ave)
#
#     loss = real((conj(transpose(xt_ave)) @ conj(transpose(A_mat)) @ A_mat @ xt_ave - 2 * real(
#         conj(transpose(zeros)) @ A_mat @ xt_ave)).item()) + 1
#     # print("Loss:", loss)
#
#     return loss
#
#
#
#
#
#     # size = decomposition_terms ** depth
#
#     # current_step = child_space[0]


# depth = 0, b
# depth = 1, U1, U2, U3


########################################################################################################################
# def expand_ansatz_tree(A, vars, ansatz_tree, backend=None, draw_tree=False, shots_power=4):
#     if backend is None:
#         backend = 'Hadamard'
#     A_coeffs = A.get_coeff()
#     A_unitaries = A.get_unitary()
#
#
#
#     A_terms_number = len(A_coeffs)
#
#     # print("Variables are:", vars)
#     # print("Ansatz Tree is:", ansatz_tree)
#
#     parent_node = ansatz_tree[-1]
#     tree_depth = len(ansatz_tree)
#     if draw_tree is True:
#         draw_ansatz_tree(A_terms_number, tree_depth, 'Expansion')
#     child_space = [parent_node + A_unitaries[i] for i in range(A_terms_number)]
#
#     if backend == 'Hadamard':
#         gradient_overlaps = [0 for _ in range(len(child_space))]
#
#         for i, child_node in enumerate(child_space):
#
#             shots = int(10 ** shots_power)
#             term_1 = 0
#             for j in range(tree_depth):
#                 term_1_1 = 0
#                 alpha = vars[j]
#                 anstaz_state = ansatz_tree[j]
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         beta_k = A_coeffs[k]
#                         beta_l = A_coeffs[l]
#                         u = U_list_dagger(child_node) + A_unitaries[k] + A_unitaries[l] + anstaz_state
#                         inner_product_real = Hadamard_test(u, shots=shots)
#                         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                         inner_product = inner_product_real - inner_product_imag * 1j
#                         term_1_1 += beta_k * beta_l * inner_product
#                 term_1 += alpha * term_1_1
#
#             term_2 = 0
#             for j in range(A_terms_number):
#                 beta_j = A_coeffs[j]
#                 u = U_list_dagger(child_node) + A_unitaries[j]
#                 inner_product_real = Hadamard_test(u, shots=shots)
#                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                 inner_product = inner_product_real - inner_product_imag * 1j
#                 term_2 += beta_j * inner_product
#
#             gradient_overlap = abs(2 * term_1 - 2 * term_2)
#
#             gradient_overlaps[i] = gradient_overlap
#
#
#         # for i, child_node in enumerate(child_space):
#         #     Shots = []
#         #     Unmiti = []
#         #     Unmiti_real = []
#         #     Unmiti_imag = []
#         #
#         #     for p in range(shots_power, 2, -1):
#         #         shots = 10 ** p
#         #         Shots.append(shots)
#         #
#         #         unbiased_exp_real = []
#         #         unbiased_exp_imag = []
#         #
#         #         unbiased_exp_number = 10
#         #         for itr in range(unbiased_exp_number):
#         #
#         #             term_1 = 0
#         #             for j in range(tree_depth):
#         #                 term_1_1 = 0
#         #                 alpha = vars[j]
#         #                 anstaz_state = ansatz_tree[j]
#         #                 for k in range(A_terms_number):
#         #                     for l in range(A_terms_number):
#         #                         beta_k = A_coeffs[k]
#         #                         beta_l = A_coeffs[l]
#         #                         u = U_list_dagger(child_node) + A_unitaries[k] + A_unitaries[l] + anstaz_state
#         #                         inner_product_real = Hadamard_test(u, shots=shots)
#         #                         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#         #                         inner_product = inner_product_real - inner_product_imag * 1j
#         #                         term_1_1 += beta_k * beta_l * inner_product
#         #                 term_1 += alpha * term_1_1
#         #
#         #             term_2 = 0
#         #             for j in range(A_terms_number):
#         #                 beta_j = A_coeffs[j]
#         #                 u = U_list_dagger(child_node) + A_unitaries[j]
#         #                 inner_product_real = Hadamard_test(u, shots=shots)
#         #                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#         #                 inner_product = inner_product_real - inner_product_imag * 1j
#         #                 term_2 += beta_j * inner_product
#         #
#         #             gradient_overlap = abs(2 * term_1 - 2 * term_2)
#         #
#         #             unbiased_exp_real.append(real(gradient_overlap))
#         #             unbiased_exp_imag.append(imag(gradient_overlap))
#         #
#         #         exp_obs_real_ave = sum(unbiased_exp_real) / unbiased_exp_number
#         #         unbiased_exp_real_half = [i for i in unbiased_exp_real if i >= exp_obs_real_ave]
#         #         unbiased_exp_real_half_exp = sum(unbiased_exp_real_half) / len(unbiased_exp_real_half)
#         #         Unmiti_real.append(unbiased_exp_real_half_exp)
#         #
#         #         exp_obs_imag_ave = sum(unbiased_exp_imag) / unbiased_exp_number
#         #         unbiased_exp_imag_half = [i for i in unbiased_exp_imag if i >= exp_obs_imag_ave]
#         #         unbiased_exp_imag_half_exp = sum(unbiased_exp_imag_half) / len(unbiased_exp_imag_half)
#         #         Unmiti_imag.append(unbiased_exp_imag_half_exp)
#         #
#         #     miti_exp_real = Richardson_extrapolate(Shots, Unmiti_real)
#         #     miti_exp_imag = Richardson_extrapolate(Shots, Unmiti_imag)
#         #     gradient_overlap = miti_exp_real + miti_exp_imag * 1j
#         #
#         #     gradient_overlaps[i] = gradient_overlap
#
#         # print("Child nodes are:", child_space)
#         # print('Estimated Gradient overlaps are:', gradient_overlaps)
#
#         # To consider the case when there are several candidates for the child node.
#         max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]
#
#         # verify_gradient_overlap(A, vars, ansatz_tree, max_index)
#
#         # candidate_states = [child_space[i] for i in max_index]
#         # print("Next child node can be chosen from:", candidate_states)
#         idx = random.choice(max_index)
#         child_node_opt = child_space[idx]
#     # print("Next child node is selected as:", child_node_opt)
#
#     # Use the matrix calculations to decide the child nodes
#     elif backend == 'Matrix':
#         A_mat = A.get_matrix()
#         x = get_x(vars, ansatz_tree)
#         zeros = zero_state()
#         width = len(A_unitaries[0][0])
#         if width > 1:
#             for j in range(width - 1):
#                 zeros = kron(zeros, zero_state())
#
#         parent_node = ansatz_tree[-1]
#         child_space = [parent_node + A_unitaries[i] for i in range(A_terms_number)]
#         gradient_overlaps = [0 for _ in range(len(child_space))]
#
#         for i, child_node in enumerate(child_space):
#             U_mat = get_unitary(child_node)
#
#             gradient = 2 * conj(transpose(zeros)) @ conj(transpose(U_mat)) @ A_mat @ A_mat @ x - 2 * conj(
#                 transpose(zeros)) @ conj(transpose(U_mat)) @ A_mat @ zeros
#             gradient_overlaps[i] = abs(gradient.item())
#
#         # print('Matrix Calculation:', gradient_overlaps)
#
#         max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]
#         idx = random.choice(max_index)
#         child_node_opt = child_space[idx]
#
#     elif backend == 'Eigens':
#         gradient_overlaps = [0 for _ in range(len(child_space))]
#         for i, child_node in enumerate(child_space):
#             term_1 = 0
#             for j in range(tree_depth):
#                 term_1_1 = 0
#                 alpha = vars[j]
#                 anstaz_state = ansatz_tree[j]
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         beta_k = A_coeffs[k]
#                         beta_l = A_coeffs[l]
#                         u = U_list_dagger(child_node) + A_unitaries[k] + A_unitaries[l] + anstaz_state
#                         inner_product = Hadamard_test(u)
#                         term_1_1 += beta_k * beta_l * inner_product
#                 term_1 += alpha * term_1_1
#
#             term_2 = 0
#             for j in range(A_terms_number):
#                 beta_j = A_coeffs[j]
#                 u = U_list_dagger(child_node) + A_unitaries[j]
#                 inner_product = Hadamard_test(u)
#                 term_2 += beta_j * inner_product
#
#             gradient_overlaps[i] = abs(2 * term_1 - 2 * term_2)
#
#         # print("Child nodes are:", child_space)
#         # print('Estimated Gradient overlaps are:', gradient_overlaps)
#
#         # To consider the case when there are several candidates for the child node.
#         max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]
#
#         # verify_gradient_overlap(A, vars, ansatz_tree, max_index)
#
#         # candidate_states = [child_space[i] for i in max_index]
#         # print("Next child node can be chosen from:", candidate_states)
#         idx = random.choice(max_index)
#         child_node_opt = child_space[idx]
#     # print("Next child node is selected as:", child_node_opt)
#
#     if draw_tree is True:
#         draw_ansatz_tree(A_terms_number, tree_depth, 'Optimal', which_index=idx)
#     ansatz_tree.append(child_node_opt)
#     if draw_tree is True:
#         draw_ansatz_tree(A_terms_number, tree_depth, 'Pending')
#     return ansatz_tree
