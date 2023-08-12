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
    Generator of the linear systems of equations problem.
"""

from typing import List, Tuple, Dict
from numpy import array, ndarray, random, pi, log2
from numpy import identity
from numpy import matmul as mat
from numpy import kron
from Error import ArgumentError
from utils import PauliStrings
from cqs_module.verifier import get_unitary
from functools import reduce

__all__ = [
    "CoeffMatrix"
]

ModuleErrorCode = 1
FileErrorCode = 0


class CoeffMatrix:
    r"""Set the ``A`` matrix.

    This class generates the coefficient matrix A of the linear system of equations with the intrinsic forms.
    It returns the unitaries, coefficients, and matrix.
    Users can also customize the A matrix with specific input.
    """

    def __init__(self, term_number, dim, width):
        r"""Set the ``A`` matrix.

        This class generates A matrix with the intrinsic forms.
        It returns the unitaries, coefficients, and matrix.
        Users can also customize the A matrix with specific input.
        """

        self.__which_form = None
        self.__unitary = None  # unitaries
        self.__matrix = None  # matrix
        self.__coeff = None  # coefficients of the unitaries
        self.__term_number = term_number
        self.__dim = dim
        # self.__width = int(log2(self.__dim))
        self.__width = width

    def generate(self, which_form=None, given_matrix=None, given_unitaries=None, given_coeffs=None):
        r"""Automatically generate a random matrix with the given intrinsic forms.

        Args:
            which_form (string, optional): choose the form to generate a matrix
            given_matrix (ndarray): customize a matrix by inputting its matrix
        """
        if which_form is None:
            which_form = "Pauli"

        # if not {which_form}.issubset(['Pauli', 'Haar']):
        #     raise ArgumentError(f"Invalid form: ({which_form})!\n"
        #                         "Only 'Pauli' and 'Haar' are supported as the "
        #                         "forms to generate the matrix. If you want to customize "
        #                         "it, please input the matrix with parameter 'given_matrix'.",
        #                         ModuleErrorCode,
        #                         FileErrorCode, 1)
        if which_form == 'Matrix':
            if given_matrix is not None:
                if isinstance(given_matrix, ndarray):
                    self.__matrix = given_matrix

                    # Check the unitary
                    # mat(self.__matrix, transpose(conj(self.__matrix))) -
                    # self.__matrix

                else:
                    raise ArgumentError(f"Invalid matrix input ({given_matrix}) with the type: `{type(given_matrix)}`!\n"
                                        "Only `ndarray` is supported as the type of the matrix.",
                                        ModuleErrorCode,
                                        FileErrorCode, 2)

        if which_form == 'Pauli':
            # Coefficients are sampled in [-2, 2] with uniform distribution
            # self.__coeff = [(random.rand() * 4 - 2) for _ in range(self.__term_number)]
            # self.__coeff = [1 for _ in range(int(self.__term_number / 2))] + [-1 for _ in range(int(self.__term_number / 2))]
            # if len(self.__coeff) != self.__term_number:
            #     self.__coeff += [random.choice([-1, 1])]
            self.__coeff = [(random.rand() * 2 - 1) for _ in range(self.__term_number)]
            # self.__coeff = [(random.rand() * 2) for _ in range(self.__term_number)]
            # self.__coeff = random.normal(0, 2, self.__term_number)


            self.__unitary = []

            # Tensor product of Pauli stings
            for i in range(self.__term_number):
                paulis = [[]]
                for j in range(self.__width):
                    paulis[0].append(random.choice(['I', 'X', 'Y', 'Z']))
                self.__unitary.append(paulis)

        elif which_form == 'Unitaries':
            self.__coeff = given_coeffs
            self.__unitary = given_unitaries

        elif which_form == 'Haar':
            self.__coeff = [(random.rand() * 2) for _ in range(self.__term_number)]
            self.__unitary = []
            for i in range(self.__term_number):
                Haar_unitary = [[]]









        self.__which_form = which_form

    # def __haar_measure(self):


    def get_unitary(self):
        return self.__unitary

    def get_coeff(self):
        return self.__coeff

    def get_width(self):
        return self.__width

    def get_matrix(self):
        mat = array([[0 for _ in range(self.__dim)] for _ in range(self.__dim)], dtype='complex128')

        for i in range(self.__term_number):
            coeff = self.__coeff[i]
            u = self.__unitary[i]
            u_mat = get_unitary(u)
            mat += coeff * u_mat

        self.__matrix = mat
        return self.__matrix


    #
    # def __sample_Pauli_strings(self, n, b):
    #     digits = [0 for _ in range(self.__width)]
    #     Pauli_stings = ['I', 'X', 'Y', 'Z']
    #     # Pauli_stings = ['X', 'X', 'X', 'X']
    #
    #     for i in range(self.__width):
    #         digits[i] = int(n % b)
    #         n //= b
    #     return [Pauli_stings[j] for j in digits[::-1]]