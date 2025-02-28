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
    Access to the efficient calculator based on Pauli eigenvalues.
"""
def U_list_dagger(U):
    return U[::-1]

def eigen_calculator(Pauli_string, basis):
    if basis == 0:
        if Pauli_string == 'X':
            basis = 1
            eigen = 1
        elif Pauli_string == 'Y':
            basis = 1
            eigen = 1j
        elif Pauli_string == 'Z':
            basis = 0
            eigen = 1
        elif Pauli_string == "I":
            basis = 0
            eigen = 1
        else:
            raise ValueError
    elif basis == 1:
        if Pauli_string == 'X':
            basis = 0
            eigen = 1
        elif Pauli_string == 'Y':
            basis = 0
            eigen = -1j
        elif Pauli_string == "Z":
            basis = 1
            eigen = -1
        elif Pauli_string == "I":
            basis = 1
            eigen = 1
        else:
            raise ValueError
    else:
        raise ValueError
    return basis, eigen

def Hadamard_test(n, U1, U2, Ub):
    # Initialize settings
    # Instead of a single unitary, we input a list of unitaries.
    # U = [U1, U2, U3] = [[[], [], ...], [[], [], ...], [[], [], ...], [[], [], ...]]
    # Each Ui = [[], [], ...] := [[column1], [column2], ...]

    # Create gates for the unitary
    # U = [U1, U2], U1 operates before U2, --> |psi> = U2 U1 |0>
    Bases = [0 for _ in range(n)]
    Eigens = [1 for _ in range(n)]
    U = U_list_dagger(Ub) + U_list_dagger(U1) + U2 + Ub
    for layer in U:
        for i, gate in enumerate(layer):
            basis, eigen = eigen_calculator(gate, Bases[i])
            Bases[i] = basis
            Eigens[i] *= eigen

    if 1 in Bases:
        return 0
    else:
        U_exp = 1
        for eigen in Eigens:
            U_exp *= eigen
        return U_exp