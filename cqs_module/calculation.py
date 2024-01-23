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
from qiskit_ionq import IonQProvider
from qiskit_braket_provider import AWSBraketProvider
import time
from datetime import datetime
from braket.aws import AwsQuantumJob

# BRAKET_DEVICE = 'SV1'
BRAKET_DEVICE = 'Aria 1'
# BRAKET_DEVICE = 'Harmony'

def U_list_dagger(U):
    return U[::-1]

# IonQ Access
# def calculate_statistics(jobs):
#     # a list of jobs
#     exps = []
#     for job in jobs:
#         if '0' not in job.get_probabilities().keys():
#             p0 = 0
#             p1 = 1
#         elif '1' not in job.get_probabilities().keys():
#             p0 = 1
#             p1 = 0
#         else:
#             p0 = job.get_probabilities()['0']
#             p1 = job.get_probabilities()['1']
#             # p0 = 0.5
#             # p1 = 0.5
#         exp = p0 - p1
#         exps.append(exp)
#     return exps

# Braket Access
def calculate_statistics(backend, jobs_ids, file_name='message.txt'):
    # a list of jobs
    exps = []
    for job_id in jobs_ids:
        if job_id == 1 or job_id == 0:
            exp = job_id
            # print("This circuit is composed of identities, skip.")
            file1 = open(file_name, "a")
            file1.writelines(["This circuit is composed of identities, skip.\n"])
            file1.close()
        else:
            job = backend.retrieve_job(job_id)
            status = job.status()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            file1 = open(file_name, "a")
            file1.writelines(["\nCurrent Time =", str(current_time), '\n'])
            file1.writelines(["Current Status:", str(status), '\n\n'])
            file1.close()
            print("Current Time =", current_time)
            print('Current Status:', status)
            print()
            DONE = status.DONE
            while status != DONE:
                time.sleep(3600)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                status = job.status()
                file1 = open(file_name, "a")
                file1.writelines(["\nCurrent Time =", str(current_time), '\n'])
                file1.writelines(["Current Status:", str(status), '\n\n'])
                file1.close()
                print("Current Time =", current_time)
                print('Status:', status)
                print()
            # while not is_result_availble():
            #     # block for a moment
            #     sleep(1)
            count = backend.retrieve_job(job_id).result().get_counts()
            new_count = {'0': 0, '1': 0}
            for k in count.keys():
                new_count[k[-1]] += count[k]
            count = new_count
            file1 = open(file_name, "a")
            file1.writelines(["The sampling result is:", str(count), '\n'])
            file1.close()
            # print("The sampling result is:", count)
            if count['0'] == 0:
                p0 = 0
                p1 = 1
            elif count['1'] == 0:
                p0 = 1
                p1 = 0
            else:
                shots = sum(list(count.values()))
                p0 = count['0'] / shots
                p1 = count['1'] / shots
            file1 = open(file_name, "a")
            file1.writelines(["The sampling probability of getting 0 is: p0 =", str(p0), '\n'])
            file1.writelines(["The sampling probability of getting 1 is: p1 =", str(p1), '\n'])
            file1.close()
            # print("The sampling probability of getting 0 is: p0 =", p0)
            # print("The sampling probability of getting 1 is: p1 =", p1)
            if count['0'] != 0 and count['1'] != 0:
                if p0 < 0.2:
                    p0 = 0
                    p1 = 1
                    file1 = open(file_name, "a")
                    file1.writelines(["p0 < 0.2: set p0 = 0 and p1 = 1.", '\n'])
                    file1.writelines(["Expectation value is -1.", '\n'])
                    file1.close()
                    # print("p0 < 0.2: set p0 = 0 and p1 = 1.")
                    # print("Expectation value is -1.")
                else:
                    if p1 < 0.2:
                        p0 = 1
                        p1 = 0
                        file1 = open(file_name, "a")
                        file1.writelines(["p0 > 0.8: set p0 = 1 and p1 = 0.", '\n'])
                        file1.writelines(["Expectation value is 1.", '\n'])
                        file1.close()
                        # print("p0 > 0.8: set p0 = 1 and p1 = 0.")
                        # print("Expectation value is 1.")
                    else:
                        p0 = 0.5
                        p1 = 0.5
                        file1 = open(file_name, "a")
                        file1.writelines(["0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.", '\n'])
                        file1.writelines(["Expectation value is 0.", '\n'])
                        file1.close()
                        # print("0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.")
                        # print("Expectation value is 0.")
            # print()
            exp = p0 - p1
        exps.append(exp)
    return exps


def calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, backend=None, shots_budget=1024, frugal=False, tasks_num = 0, shots_num = 0, file_name='message.txt'):
    """
        Please note that the objective function of CVXOPT has the form:   1/2  x^T P x  +   q^T x
        But our objective function is:                                         z^T Q z  - 2 r^T z + 1
        So the coefficients are corrected here.

    :param R:
    :param I:
    :param q:
    :return: Q, r
    """
    A_coeffs = A.get_coeff()
    A_unitaries = A.get_unitary()
    A_terms_number = len(A_coeffs)
    tree_depth = len(ansatz_tree)
    V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')

    if backend is None:
        backend = 'eigens'

    if frugal is True:
        shots_each_entry = shots_budget / (10 * (tree_depth ** 2 + tree_depth))
        M_A_A = sum(abs(conj(A_coeffs[k]) * A_coeffs[l])
                    for k in range(A_terms_number) for l in range(A_terms_number))
        P_A_A = [10 * int(shots_each_entry * (abs(conj(A_coeffs[k]) * A_coeffs[l]) / M_A_A))
                 for k in range(A_terms_number) for l in range(A_terms_number)]
        M_A = sum([abs(conj(A_coeffs[k]))
                   for k in range(A_terms_number)])
        P_A = [10 * int(shots_each_entry * (abs(conj(A_coeffs[k])) / M_A)) for k in range(A_terms_number)]

    else:
        P_A_A = [100 for _ in range(A_terms_number) for _ in range(A_terms_number)]
        P_A = [100 for _ in range(A_terms_number)]
    #
    # U_K = []
    # for i in range(tree_depth):
    #     for j in range(tree_depth):
    #         # Uniform distribution of the shots
    #         for k in range(A_terms_number):
    #             for l in range(A_terms_number):
    #                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
    #                 U_K.append(u)
    #
    # U_q = []
    # for i in range(tree_depth):
    #     for k in range(A_terms_number):
    #         u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
    #         U_q.append(u)
    #
    # U = U_K + U_q
    # C_R = Hadamard_test(U, backend='ionq', alpha=1)
    # C_I = Hadamard_test(U, backend='ionq', alpha=1j)
    #
    # C = C_R + C_I
    # provider = IonQProvider('pUhwyKCHRYAvWUChFqwTApQwow4mS2h7')
    # simulator_backend = provider.get_backend("ionq_qpu.harmony")
    # jobs = execute(C, backend=simulator_backend, shots=100)
    # print("job id:", jobs.job_id)
    #
    # C_exp = []
    # for job in jobs:
    #     if '0' not in job.get_probabilities().keys():
    #         p0 = 0
    #         p1 = 1
    #     elif '1' not in job.get_probabilities().keys():
    #         p0 = 1
    #         p1 = 0
    #     else:
    #         p0 = job.get_probabilities()['0']
    #         p1 = job.get_probabilities()['1']
    #     real_exp = p0 - p1
    # #
    #
    if backend == 'ionq' or backend == 'braket':
        Job_ids_K_R = []
        Job_ids_K_I = []
        Job_ids_q_R = []
        Job_ids_q_I = []
        for i in range(tree_depth):
            for j in range(tree_depth):
                # Uniform distribution of the shots
                item = 0
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
                        shots = P_A_A[k * A_terms_number + l]
                        shots = 20
                        # file1 = open(file_name, "a")
                        # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                        # file1.close()
                        jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        Job_ids_K_R.append(jobid_R)
                        Job_ids_K_I.append(jobid_I)
        for i in range(tree_depth):
            for k in range(A_terms_number):
                u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
                shots = P_A[k]
                shots = 20
                # file1 = open(file_name, "a")
                # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                # file1.close()
                jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                Job_ids_q_R.append(jobid_R)
                Job_ids_q_I.append(jobid_I)

        if backend == 'ionq':
            provider = IonQProvider('pUhwyKCHRYAvWUChFqwTApQwow4mS2h7')
            # simulator_backend = provider.get_backend("ionq_qpu.harmony")
            # simulator_backend = provider.get_backend("ionq_qpu.aria-1")
            simulator_backend = provider.get_backend("ionq_simulator")
        else:
            provider = AWSBraketProvider()
            simulator_backend = provider.get_backend(BRAKET_DEVICE)

        exp_K_R = calculate_statistics(simulator_backend, Job_ids_K_R, file_name=file_name)
        exp_K_I = calculate_statistics(simulator_backend, Job_ids_K_I, file_name=file_name)
        exp_q_R = calculate_statistics(simulator_backend, Job_ids_q_R, file_name=file_name)
        exp_q_I = calculate_statistics(simulator_backend, Job_ids_q_I, file_name=file_name)

        for i in range(tree_depth):
            for j in range(tree_depth):
                # Uniform distribution of the shots
                item = 0
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        inner_product_real = exp_K_R[i * tree_depth * A_terms_number * A_terms_number +
                                                     j * A_terms_number * A_terms_number +
                                                     k * A_terms_number +
                                                     l]
                        inner_product_imag = exp_K_I[i * tree_depth * A_terms_number * A_terms_number +
                                                     j * A_terms_number * A_terms_number +
                                                     k * A_terms_number +
                                                     l]
                        inner_product = inner_product_real - inner_product_imag * 1j
                        item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
                V_dagger_V[i][j] = item

        R = real(V_dagger_V)
        I = imag(V_dagger_V)

        q = zeros((tree_depth, 1), dtype='complex128')
        for i in range(tree_depth):
            item = 0
            for k in range(A_terms_number):
                inner_product_real = exp_q_R[i * A_terms_number + k]
                inner_product_imag = exp_q_I[i * A_terms_number + k]
                inner_product = inner_product_real - inner_product_imag * 1j
                item += conj(A_coeffs[k]) * inner_product
            q[i][0] = item

    else:
        for i in range(tree_depth):
            for j in range(tree_depth):
                item = 0
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
                        shots = P_A_A[k * A_terms_number + l]
                        shots = 20
                        # file1 = open(file_name, "a")
                        # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                        # file1.close()
                        inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        inner_product = inner_product_real - inner_product_imag * 1j
                        item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
                V_dagger_V[i][j] = item

        R = real(V_dagger_V)
        I = imag(V_dagger_V)

        q = zeros((tree_depth, 1), dtype='complex128')
        for i in range(tree_depth):
            item = 0
            for k in range(A_terms_number):
                u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
                shots = P_A[k]
                shots = 20
                # file1 = open(file_name, "a")
                # file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                # file1.close()
                inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                inner_product = inner_product_real - inner_product_imag * 1j
                item += conj(A_coeffs[k]) * inner_product
            q[i][0] = item

    # Q     =      R    -I
    #       =      I     R
    Q = array(append(append(R, -I, axis=1), append(I, R, axis=1), axis=0), dtype='float64')

    # r = [Re(q),
    #      Im(q)]
    r_real = real(q)
    r_imag = imag(q)
    r = array(append(r_real, r_imag, axis=0), dtype='float64')
    return Q, r, tasks_num, shots_num


def calculate_loss_function(A, vars, ansatz_tree, backend=None, shots_budget=1024, frugal=False, tasks_num = 0, shots_num = 0, file_name='message.txt'):
    A_coeffs = A.get_coeff()
    A_unitaries = A.get_unitary()
    A_terms_number = len(A_coeffs)
    tree_depth = len(ansatz_tree)

    if backend is None:
        backend = 'eigens'

    if frugal is True:
        shots_budget = shots_budget / 10
        M_Loss = sum([abs(conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l])
                      for i in range(tree_depth)
                      for j in range(tree_depth)
                      for k in range(A_terms_number)
                      for l in range(A_terms_number)] +
                     [2 * abs(real(vars[i]) * A_coeffs[j])
                      for i in range(tree_depth)
                      for j in range(A_terms_number)] +
                     [2 * abs(imag(vars[i]) * A_coeffs[j])
                      for i in range(tree_depth)
                      for j in range(A_terms_number)]
                     )
        P_Loss_term_1 = [
            10 * int(shots_budget * abs(conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l]) / M_Loss)
            for i in range(tree_depth)
            for j in range(tree_depth)
            for k in range(A_terms_number)
            for l in range(A_terms_number)]
        P_Loss_term_2 = [10 * int(shots_budget * 2 * abs(real(vars[i]) * A_coeffs[j]) / M_Loss)
                         for i in range(tree_depth)
                         for j in range(A_terms_number)]
        P_Loss_term_3 = [10 * int(shots_budget * 2 * abs(imag(vars[i]) * A_coeffs[j]) / M_Loss)
                         for i in range(tree_depth)
                         for j in range(A_terms_number)]

    else:
        P_Loss_term_1 = [100 for _ in range((tree_depth ** 2) * (A_terms_number ** 2))]
        P_Loss_term_2 = [100 for _ in range(tree_depth * A_terms_number)]
        P_Loss_term_3 = [100 for _ in range(tree_depth * A_terms_number)]

    if backend == 'ionq' or backend == 'braket':
        Job_ids_1_R = []
        Job_ids_1_I = []
        Job_ids_2_R = []
        Job_ids_3_I = []
        term_1 = 0
        for i in range(tree_depth):
            for j in range(tree_depth):
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        shots = P_Loss_term_1[i * tree_depth * A_terms_number * A_terms_number +
                                              j * A_terms_number * A_terms_number +
                                              k * A_terms_number +
                                              l]
                        shots = 20
                        u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
                            ansatz_tree[j]
                        file1 = open(file_name, "a")
                        file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                        file1.close()
                        jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        Job_ids_1_R.append(jobid_R)
                        jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        Job_ids_1_I.append(jobid_I)

        for i in range(tree_depth):
            for j in range(A_terms_number):
                shots = P_Loss_term_2[i * A_terms_number + j]
                shots = 20
                u = A_unitaries[j] + ansatz_tree[i]
                file1 = open(file_name, "a")
                file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                file1.close()
                jobid_R, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                Job_ids_2_R.append(jobid_R)

        for i in range(tree_depth):
            for j in range(A_terms_number):
                shots = P_Loss_term_3[i * A_terms_number + j]
                shots = 20
                u = A_unitaries[j] + ansatz_tree[i]
                file1 = open(file_name, "a")
                file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                file1.close()
                jobid_I, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                Job_ids_3_I.append(jobid_I)


        if backend == 'ionq':
            provider = IonQProvider('pUhwyKCHRYAvWUChFqwTApQwow4mS2h7')
            # simulator_backend = provider.get_backend("ionq_qpu.harmony")
            # simulator_backend = provider.get_backend("ionq_qpu.aria-1")
            simulator_backend = provider.get_backend("ionq_simulator")
        else:
            provider = AWSBraketProvider()
            simulator_backend = provider.get_backend(BRAKET_DEVICE)

        exp_1_R = calculate_statistics(simulator_backend, Job_ids_1_R)
        exp_1_I = calculate_statistics(simulator_backend, Job_ids_1_I)
        exp_2_R = calculate_statistics(simulator_backend, Job_ids_2_R)
        exp_3_I = calculate_statistics(simulator_backend, Job_ids_3_I)


        term_1 = 0
        for i in range(tree_depth):
            for j in range(tree_depth):
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        inner_product_real = exp_1_R[i * tree_depth * A_terms_number * A_terms_number +
                                                  j * A_terms_number * A_terms_number +
                                                  k * A_terms_number +
                                                  l]
                        inner_product_imag = exp_1_I[i * tree_depth * A_terms_number * A_terms_number +
                                                  j * A_terms_number * A_terms_number +
                                                  k * A_terms_number +
                                                  l]
                        inner_product = inner_product_real - inner_product_imag * 1j
                        term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product

        term_2 = 0
        for i in range(tree_depth):
            for j in range(A_terms_number):
                inner_product_real = exp_2_R[i * A_terms_number + j]
                term_2 += real(vars[i]) * A_coeffs[j] * inner_product_real

        term_3 = 0
        for i in range(tree_depth):
            for j in range(A_terms_number):
                inner_product_imag = exp_3_I[i * A_terms_number + j]
                term_3 += imag(vars[i]) * A_coeffs[j] * inner_product_imag

    else:
        term_1 = 0
        for i in range(tree_depth):
            for j in range(tree_depth):
                for k in range(A_terms_number):
                    for l in range(A_terms_number):
                        shots = P_Loss_term_1[i * tree_depth * A_terms_number * A_terms_number +
                                              j * A_terms_number * A_terms_number +
                                              k * A_terms_number +
                                              l]
                        shots = 20
                        u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
                            ansatz_tree[j]
                        file1 = open(file_name, "a")
                        file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                        file1.close()
                        inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                        inner_product = inner_product_real - inner_product_imag * 1j
                        term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product

        term_2 = 0
        for i in range(tree_depth):
            for j in range(A_terms_number):
                shots = P_Loss_term_2[i * A_terms_number + j]
                shots = 20
                u = A_unitaries[j] + ansatz_tree[i]
                file1 = open(file_name, "a")
                file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                file1.close()
                inner_product_real, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                term_2 += real(vars[i]) * A_coeffs[j] * inner_product_real

        term_3 = 0
        for i in range(tree_depth):
            for j in range(A_terms_number):
                shots = P_Loss_term_3[i * A_terms_number + j]
                shots = 20
                u = A_unitaries[j] + ansatz_tree[i]
                file1 = open(file_name, "a")
                file1.writelines(["The unitary for estimation is:", str(u), '\n'])
                file1.close()
                inner_product_imag, tasks_num, shots_num = Hadamard_test(u, backend=backend, alpha=1j, shots=shots, tasks_num = tasks_num, shots_num = shots_num)
                term_3 += imag(vars[i]) * A_coeffs[j] * inner_product_imag

    loss = term_1 - 2 * (term_2 + term_3) + 1
    return loss, tasks_num, shots_num

########################################################################################################################


# def calculate_Q_r_by_Hadamrd_test(A, ansatz_tree, mtd=None, shots_power=4):
#     """
#         Please note that the objective function of CVXOPT has the form:   1/2  x^T P x  +   q^T x
#         But our objective function is:                                         z^T Q z  - 2 r^T z + 1
#         So the coefficients are corrected here.
#
#     :param R:
#     :param I:
#     :param q:
#     :return: Q, r
#     """
#     A_coeffs = A.get_coeff()
#     A_unitaries = A.get_unitary()
#     A_terms_number = len(A_coeffs)
#
#     tree_depth = len(ansatz_tree)
#     V_dagger_V = zeros((tree_depth, tree_depth), dtype='complex128')
#
#     if mtd is None:
#         mtd = 'Hadamard'
#
#     for i in range(tree_depth):
#         for j in range(tree_depth):
#             # Unmiti = []
#             # Unmiti_real = []
#             # Unmiti_imag = []
#
#             shots = int(10 ** shots_power)
#             item = 0
#             # print_progress((tree_depth * i + j) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#             for k in range(A_terms_number):
#                 for l in range(A_terms_number):
#                     u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#                     if mtd == 'Hadamard':
#                         inner_product_real = Hadamard_test(u, shots=shots)
#                         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                         inner_product = inner_product_real - inner_product_imag * 1j
#
#                     elif mtd == 'Matrix':
#                         inner_product = Hadmard_test_by_matrix(u)
#
#                     elif mtd == 'Eigens':
#                         inner_product = Hadamard_test(u)
#
#                     item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#
#             V_dagger_V[i][j] = item
#
#             # Shots = []
#             # Unmiti = []
#             # Unmiti_real = []
#             # Unmiti_imag = []
#             #
#             # for p in range(shots_power, 2, -1):
#             #     shots = 10 ** p
#             #     Shots.append(shots)
#             #     unbiased_exp_real = []
#             #     unbiased_exp_imag = []
#             #
#             #     unbiased_exp_number = 10
#             #     for itr in range(unbiased_exp_number):
#             #         #
#             #         #     item = 0
#             #         #     # print_progress((tree_depth * i + j) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#             #         #     for k in range(A_terms_number):
#             #         #         for l in range(A_terms_number):
#             #         #             u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#             #         #             if mtd == 'Hadamard':
#             #         #                 inner_product_real = Hadamard_test(u, shots=shots)
#             #         #                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#             #         #                 inner_product = inner_product_real - inner_product_imag * 1j
#             #         #
#             #         #             elif mtd == 'Matrix':
#             #         #                 inner_product = Hadmard_test_by_matrix(u)
#             #         #
#             #         #             elif mtd == 'Eigens':
#             #         #                 inner_product = Hadamard_test(u)
#             #         #
#             #         #             item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#             #         #
#             #         #     unbiased_exp.append(item)
#             #         #
#             #         # exp_obs_ave = sum(unbiased_exp) / unbiased_exp_number
#             #         # unbiased_exp_half = [i for i in unbiased_exp if i >= exp_obs_ave]
#             #         # unbiased_exp_half_exp = sum(unbiased_exp_half) / len(unbiased_exp_half)
#             #         # # if exp_obs_ave > real_exp_std:
#             #         # Shots.append(shot)
#             #         # Exp.append(unbiased_exp_half_exp)
#             #
#             #         item = 0
#             #         # print_progress((tree_depth * i + j) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#             #         for k in range(A_terms_number):
#             #             for l in range(A_terms_number):
#             #                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
#             #                     ansatz_tree[j]
#             #                 if mtd == 'Hadamard':
#             #                     inner_product_real = Hadamard_test(u, shots=shots)
#             #                     inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#             #                     inner_product = inner_product_real - inner_product_imag * 1j
#             #
#             #                 elif mtd == 'Matrix':
#             #                     inner_product = Hadmard_test_by_matrix(u)
#             #
#             #                 elif mtd == 'Eigens':
#             #                     inner_product = Hadamard_test(u)
#             #
#             #                 item += conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#             #
#             #         unbiased_exp_real.append(real(item))
#             #         unbiased_exp_imag.append(imag(item))
#             #
#             #     exp_obs_real_ave = sum(unbiased_exp_real) / unbiased_exp_number
#             #     unbiased_exp_real_half = [i for i in unbiased_exp_real if i >= exp_obs_real_ave]
#             #     unbiased_exp_real_half_exp = sum(unbiased_exp_real_half) / len(unbiased_exp_real_half)
#             #     Unmiti_real.append(unbiased_exp_real_half_exp)
#             #
#             #     exp_obs_imag_ave = sum(unbiased_exp_imag) / unbiased_exp_number
#             #     unbiased_exp_imag_half = [i for i in unbiased_exp_imag if i >= exp_obs_imag_ave]
#             #     unbiased_exp_imag_half_exp = sum(unbiased_exp_imag_half) / len(unbiased_exp_imag_half)
#             #     Unmiti_imag.append(unbiased_exp_imag_half_exp)
#             #
#             # miti_exp_real = Richardson_extrapolate(Shots, Unmiti_real)
#             # miti_exp_imag = Richardson_extrapolate(Shots, Unmiti_imag)
#             #
#             # miti_exp = miti_exp_real + miti_exp_imag * 1j
#             # V_dagger_V[i][j] = miti_exp
#
#
#
#
#
#
#
#
#
#     R = real(V_dagger_V)
#     I = imag(V_dagger_V)
#
#     q = zeros((tree_depth, 1), dtype='complex128')
#
#     for i in range(tree_depth):
#         shots = int(10 ** shots_power)
#
#         item = 0
#         # print_progress((tree_depth * tree_depth + i + 1) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#         for k in range(A_terms_number):
#             u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#             if mtd == 'Hadamard':
#                 inner_product_real = Hadamard_test(u, shots=shots)
#                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                 inner_product = inner_product_real - inner_product_imag * 1j
#
#             elif mtd == 'Matrix':
#                 inner_product = Hadmard_test_by_matrix(u)
#
#             elif mtd == 'Eigens':
#                 inner_product = Hadamard_test(u)
#
#             item += conj(A_coeffs[k]) * inner_product
#
#         q[i][0] = item
#
#
#
#
#
#
#     # for i in range(tree_depth):
#     #     Shots = []
#     #     Unmiti = []
#     #     Unmiti_real = []
#     #     Unmiti_imag = []
#     #
#     #     for p in range(shots_power, 2, -1):
#     #         shots = 10 ** p
#     #         Shots.append(shots)
#     #
#     #         unbiased_exp_real = []
#     #         unbiased_exp_imag = []
#     #
#     #         unbiased_exp_number = 10
#     #         for itr in range(unbiased_exp_number):
#     #
#     #             item = 0
#     #             # print_progress((tree_depth * tree_depth + i + 1) / ((tree_depth * tree_depth) + tree_depth), 'Current Progress:')
#     #             for k in range(A_terms_number):
#     #                 u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k])
#     #                 if mtd == 'Hadamard':
#     #                     inner_product_real = Hadamard_test(u, shots=shots)
#     #                     inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#     #                     inner_product = inner_product_real - inner_product_imag * 1j
#     #
#     #                 elif mtd == 'Matrix':
#     #                     inner_product = Hadmard_test_by_matrix(u)
#     #
#     #                 elif mtd == 'Eigens':
#     #                     inner_product = Hadamard_test(u)
#     #
#     #                 item += conj(A_coeffs[k]) * inner_product
#     #
#     #             unbiased_exp_real.append(real(item))
#     #             unbiased_exp_imag.append(imag(item))
#     #
#     #         exp_obs_real_ave = sum(unbiased_exp_real) / unbiased_exp_number
#     #         unbiased_exp_real_half = [i for i in unbiased_exp_real if i >= exp_obs_real_ave]
#     #         unbiased_exp_real_half_exp = sum(unbiased_exp_real_half) / len(unbiased_exp_real_half)
#     #         Unmiti_real.append(unbiased_exp_real_half_exp)
#     #
#     #         exp_obs_imag_ave = sum(unbiased_exp_imag) / unbiased_exp_number
#     #         unbiased_exp_imag_half = [i for i in unbiased_exp_imag if i >= exp_obs_imag_ave]
#     #         unbiased_exp_imag_half_exp = sum(unbiased_exp_imag_half) / len(unbiased_exp_imag_half)
#     #         Unmiti_imag.append(unbiased_exp_imag_half_exp)
#     #
#     #
#     #     miti_exp_real = Richardson_extrapolate(Shots, Unmiti_real)
#     #     miti_exp_imag = Richardson_extrapolate(Shots, Unmiti_imag)
#     #
#     #     miti_exp = miti_exp_real + miti_exp_imag * 1j
#     #
#     #
#     #     q[i][0] = miti_exp
#
#
#
#     # Q     =      R    -I
#     #       =      I     R
#     Q = array(append(append(R, -I, axis=1), append(I, R, axis=1), axis=0), dtype='float64')
#
#     # r = [Re(q),
#     #      Im(q)]
#     r_real = real(q)
#     r_imag = imag(q)
#     r = array(append(r_real, r_imag, axis=0), dtype='float64')
#     return Q, r


# def calculate_loss_function(A, vars, ansatz_tree, mtd=None, shots_power=4):
#     A_coeffs = A.get_coeff()
#     A_unitaries = A.get_unitary()
#     A_terms_number = len(A_coeffs)
#     tree_depth = len(ansatz_tree)
#
#     if mtd is None:
#         mtd = 'Hadamard'
#
#     if mtd == 'Hadamard':
#
#         shots = int(10 ** shots_power)
#
#
#         term_1 = 0
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#                         inner_product_real = Hadamard_test(u, shots=shots)
#                         inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                         inner_product = inner_product_real - inner_product_imag * 1j
#                         term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#
#         term_2 = 0
#         for i in range(tree_depth):
#             for j in range(A_terms_number):
#                 u = A_unitaries[j] + ansatz_tree[i]
#                 inner_product_real = Hadamard_test(u, shots=shots)
#                 inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#                 inner_product = inner_product_real - inner_product_imag * 1j
#                 term_2 += real(vars[i] * A_coeffs[j] * inner_product)
#         loss = term_1 - 2 * term_2 + 1
#
#         # if mtd == 'Hadamard':
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
#         #             for i in range(tree_depth):
#         #                 for j in range(tree_depth):
#         #                     for k in range(A_terms_number):
#         #                         for l in range(A_terms_number):
#         #                             u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + \
#         #                                 ansatz_tree[j]
#         #                             inner_product_real = Hadamard_test(u, shots=shots)
#         #                             inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#         #                             inner_product = inner_product_real - inner_product_imag * 1j
#         #                             term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#         #
#         #             term_2 = 0
#         #             for i in range(tree_depth):
#         #                 for j in range(A_terms_number):
#         #                     u = A_unitaries[j] + ansatz_tree[i]
#         #                     inner_product_real = Hadamard_test(u, shots=shots)
#         #                     inner_product_imag = Hadamard_test(u, alpha=1j, shots=shots)
#         #                     inner_product = inner_product_real - inner_product_imag * 1j
#         #                     term_2 += real(vars[i] * A_coeffs[j] * inner_product)
#         #             loss = term_1 - 2 * term_2 + 1
#         #
#         #             unbiased_exp_real.append(real(loss))
#         #             unbiased_exp_imag.append(imag(loss))
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
#         #
#         #     loss = miti_exp_real + miti_exp_imag * 1j
#
#
#
#     elif mtd == 'Matrix':
#         A_mat = A.get_matrix()
#         x = get_x(vars, ansatz_tree)
#         zeros = zero_state()
#         width = len(A_unitaries[0][0])
#         if width > 1:
#             for j in range(width - 1):
#                 zeros = kron(zeros, zero_state())
#         loss = real((conj(transpose(x)) @ conj(transpose(A_mat)) @ A_mat @ x - 2 * real(conj(transpose(zeros)) @ A_mat @ x)).item()) + 1
#
#     elif mtd == "Eigens":
#         term_1 = 0
#         for i in range(tree_depth):
#             for j in range(tree_depth):
#                 for k in range(A_terms_number):
#                     for l in range(A_terms_number):
#                         u = U_list_dagger(ansatz_tree[i]) + U_list_dagger(A_unitaries[k]) + A_unitaries[l] + ansatz_tree[j]
#                         inner_product = Hadamard_test(u)
#                         term_1 += conj(vars[i]) * vars[j] * conj(A_coeffs[k]) * A_coeffs[l] * inner_product
#
#         term_2 = 0
#         for i in range(tree_depth):
#             for j in range(A_terms_number):
#                 u = A_unitaries[j] + ansatz_tree[i]
#                 inner_product = Hadamard_test(u)
#                 term_2 += real(vars[i] * A_coeffs[j] * inner_product)
#         loss = term_1 - 2 * term_2 + 1
#
#     return loss
