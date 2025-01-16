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
    This is the module to record all the classes and objects in CQS.
    First, it has the generator of the linear systems of equations problem.
"""

from numpy import zeros
from numpy import abs as np_abs
from numpy import sum as np_sum
from random import choice, random
from Error import ArgumentError
from cqs.verifier import get_unitary
from typing import List
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator

__all__ = [
    "Instance",
    "RandomInstance"
]

ModuleErrorCode = 1
FileErrorCode = 0


class Instance:
    r"""Set the A matrix and unitary for b.

    This class generates the coefficient matrix A and unitary b of the linear system of equations
    according to the corresponding inputs.
    Users can also customize the A matrix with specific input.
    """

    def __init__(self, n, K):
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
        self.__dim = 2 ** n

    def generate(self, given_coeffs=None, given_unitaries=None, given_ub=None):
        r"""Automatically generate a random matrix with the given intrinsic forms.

        Args:
            given_coeffs (List): a list of coefficients
            given_unitaries (List): a list of unitaries
            given_ub (List): a unitary that corresponds to the state |b>
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

    def __calculate_matrix(self):
        shape = (self.__dim, self.__dim)
        A_mat = zeros(shape, dtype='complex128')
        for i in range(self.__num_term):
            u = self.__unitaries[i]
            if type(u) is list:
                u_mat = get_unitary(u)
            else:
                u_mat = Operator(u).data
            c = self.__coeffs[i]
            A_mat += c * u_mat
        self.__matrix = A_mat

    def get_matrix(self):
        self.__calculate_matrix()
        return self.__matrix

class RandomInstance():
    r"""Set the A matrix and unitary for b.

    This class generates the coefficient matrix A and unitary b of the linear system of equations with the intrinsic forms.
    It returns the unitaries, coefficients, and matrix.
    Users can also customize the A matrix with specific input and properties such as invertibility.
    """

    def __init__(self, n, K):
        r"""Set the A matrix and unitary for b.

        This class generates the coefficient matrix A and unitary b of the linear system of equations with the intrinsic forms.
        It returns the unitaries, coefficients, and matrix.
        Users can also customize the A matrix with specific input and properties such as invertibility.

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
        self.__dim = 2 ** n

    def __generate_random_Pauli(self):
        # Tensor product of Pauli stings
        while True:
            random_Pauli = [[]]
            for j in range(self.__num_qubit):
                random_Pauli[0].append(choice(['I', 'X', 'Y', 'Z']))
            counter = 0
            for p in random_Pauli[0]:
                if p in ['I', 'Z']:
                    counter += 1
            if 0 <= counter < len(random_Pauli[0]):
                break
        return random_Pauli

    def __Pauli_to_gate(self, u):
        qr = QuantumRegister(self.__num_qubit, 'q')
        cr = ClassicalRegister(self.__num_qubit, 'c')
        cir = QuantumCircuit(qr, cr)
        for i, p in enumerate(u[0]):
            if p == 'I':
                cir.id(qr[i])
            elif p == 'X':
                cir.x(qr[i])
            elif p == 'Y':
                cir.y(qr[i])
            elif p == 'Z':
                cir.z(qr[i])
            else:
                raise ValueError
        return cir

    def __calculate_matrix_pre(self):
        shape = (self.__dim, self.__dim)
        A_mat_pre = zeros(shape, dtype='complex128')
        for i in range(self.__num_term - 1):
            u = self.__unitaries[i]
            if type(u) is list:
                u_mat = get_unitary(u)
            else:
                u_mat = Operator(u).data
            c = self.__coeffs[i]
            A_mat_pre += c * u_mat
        return A_mat_pre

    def __calculate_diagonal_pre(self, A_mat_pre):
        diag_pre_max = max(np_sum(np_abs(A_mat_pre), axis=1)) + 1
        return diag_pre_max

    def __generate_identity(self):
        A_mat_pre = self.__calculate_matrix_pre()
        diag_pre_max = self.__calculate_diagonal_pre(A_mat_pre)
        self.__coeffs.append(diag_pre_max)
        self.__coeffs = self.__coeffs / diag_pre_max
        if self.__which_type in ['Pauli_eigens', 'Pauli_gates']:
            self.__unitaries.append([['I' for _ in range(self.__num_qubit)]])
        elif self.__which_type in ['Haar']:
            self.__unitaries.append(self.__Pauli_to_gate([['I' for _ in range(self.__num_qubit)]]))

    def __generate_random_unitary(self):
        return [random_circuit(self.__num_qubit, self.__num_qubit, measure=False) for _ in range(self.__num_term - 1)]

    def generate(self, which_type=None, given_ub=None):
        r"""Automatically generate a random matrix with the given intrinsic forms.

        Args:
            which_type(string, optional): choose the form to generate a matrix.
                                        If None, "Pauli_eigens" will be set as default;
            given_ub (optional): a unitary that corresponds to the state |b>.
                                        If None, a randomized ub will be set as default;
        """
        if which_type is None:
            self.__which_type = "Pauli_eigens"
        else:
            self.__which_type = which_type

        if self.__which_type == 'Pauli_eigens':
            # We further make sure that the created matrix A is Hermitian and invertible
            self.__coeffs = [(random() * 2 - 1) for _ in range(self.__num_term - 1)]
            self.__unitaries = []
            for i in range(self.__num_term - 1):
                random_Pauli = self.__generate_random_Pauli()
                self.__unitaries.append(random_Pauli)
            self.__generate_identity()
            if given_ub is None:
                self.__ub = self.__generate_random_Pauli()
            else:
                assert type(given_ub) is list
                self.__ub = given_ub

        elif self.__which_type == 'Pauli_gates':
            # We further make sure that the created matrix A is Hermitian and invertible
            self.__coeffs = [(random() * 2 - 1) for _ in range(self.__num_term - 1)]
            self.__unitaries = []
            for i in range(self.__num_term - 1):
                random_Pauli = self.__generate_random_Pauli()
                self.__unitaries.append(random_Pauli)
            self.__generate_identity()
            for i in range(self.__num_term):
                u = self.__unitaries[i]
                cir = self.__Pauli_to_gate(u)
                self.__unitaries[i] = cir
            if given_ub is None:
                self.__ub = self.__Pauli_to_gate(self.__generate_random_Pauli())
            else:
                assert type(given_ub) is QuantumCircuit
                self.__ub = given_ub

        elif self.__which_type == 'Haar':
            self.__coeffs = [(random() * 2 - 1) for _ in range(self.__num_term - 1)]
            self.__unitaries = self.__generate_random_unitary()
            self.__generate_identity()
            if given_ub is None:
                self.__ub = random_circuit(self.__num_qubit, self.__num_qubit, measure=False)
            else:
                assert type(given_ub) is QuantumCircuit
                self.__ub = given_ub
        else:
            raise ValueError

    def get_unitaries(self):
        return self.__unitaries

    def get_coeffs(self):
        return self.__coeffs

    def get_num_qubit(self):
        return self.__num_qubit

    def get_input_type(self):
        return self.__which_type

    def get_ub(self):
        return self.__ub

    def __calculate_matrix(self):
        shape = (self.__dim, self.__dim)
        A_mat = zeros(shape, dtype='complex128')
        for i in range(self.__num_term):
            u = self.__unitaries[i]
            if type(u) is list:
                u_mat = get_unitary(u)
            else:
                u_mat = Operator(u).data
            c = self.__coeffs[i]
            A_mat += c * u_mat
        self.__matrix = A_mat

    def get_matrix(self):
        self.__calculate_matrix()
        return self.__matrix

