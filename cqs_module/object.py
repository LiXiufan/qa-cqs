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

from typing import List, Tuple, Dict
from numpy import array, ndarray, random
from Error import ArgumentError
from cqs_module.verifier import get_unitary

import time
from datetime import datetime

import numpy as np
from typing import Union, Tuple, Dict
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.providers import JobStatus
import logging


__all__ = [
    "CoeffMatrix",
    "InnerProduct"
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

        if which_form == 'Matrix':
            if given_matrix is not None:
                if isinstance(given_matrix, ndarray):
                    self.__matrix = given_matrix

                else:
                    raise ArgumentError(
                        f"Invalid matrix input ({given_matrix}) with the type: `{type(given_matrix)}`!\n"
                        "Only `ndarray` is supported as the type of the matrix.",
                        ModuleErrorCode,
                        FileErrorCode, 2)

        if which_form == 'Pauli':
            self.__coeff = [(random.rand() * 2 - 1) for _ in range(self.__term_number)]
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


class InnerProduct():
    r"""Set the inner product class.

    This class records the inner products used for calculating the auxiliary systems Q and r.
    In our code implementation, we pseudo-execute all inner products at the first step and
    record the instances with this class. Then we submitted all instances to compute the inner products
    and store their values into instances of this class. Then we can obtain the values by indexes.

    Attributes:
        access (str): different access to the backend
        b (Union[np.ndarray, QuantumCircuit, Tuple[Dict[int, complex], int]]): quantum circuit for preparing b
        term_number (int): number of decomposition terms
        threshold (int): truncated threshold of our algorithm
        shots (int, optional): number of measurements
    """

    def __init__(self, access: str, b: Union[np.ndarray, QuantumCircuit, Tuple[Dict[int, complex], int]],
                 term_number: int, threshold: int, shots: int = 1024):
        r"""Set the inner product class.

        This class records the inner products used for calculating the auxiliary systems Q and r.
        In our code implementation, we pseudo-execute all inner products at the first step and
        record the instances with this class. Then we submitted all instances to compute the inner products
        and store their values into instances of this class. Then we can obtain the values by indexes.

        Args:
            access (str): different access to the backend
            b (Union[np.ndarray, QuantumCircuit, Tuple[Dict[int, complex], int]]): quantum circuit for preparing b
            term_number (int): number of decomposition terms
            threshold (int): truncation threshold of our algorithm
            shots (int, optional): number of measurements
        """
        self.access = access
        self.shots = shots
        self.b = b
        self.pos_inner_product_real = np.empty(self.power, dtype=np.float64)
        self.pos_inner_product_imag = np.empty(self.power, dtype=np.float64)
        self.neg_inner_product_real = np.empty(self.power, dtype=np.float64)
        self.neg_inner_product_imag = np.empty(self.power, dtype=np.float64)
        self.non_q = ["true", "sample", "sparse"]
        if self.access not in self.non_q:
            self.backend = get_backend(self.access)









        # self._calculate_inner_product()

    # def get_inner_product(self, q_pow: int, imag: bool = False):
    #     r"""Get the value of an inner product.
    #
    #     Args:
    #         q_pow (int): the power of permutation matrix
    #         imag (bool, optional): False: calculate the real part;
    #                                True: calculate the imaginary part
    #
    #     Returns:
    #         float / int: the value of an inner product
    #     """
    #     if q_pow == 0 and not imag:
    #         return 1
    #     elif q_pow == 0 and imag:
    #         return 0
    #     elif q_pow > 0 and not imag:
    #         return self.pos_inner_product_real[q_pow - 1]
    #     elif q_pow > 0 and imag:
    #         return self.pos_inner_product_imag[q_pow - 1]
    #     elif q_pow < 0 and not imag:
    #         return self.neg_inner_product_real[-(q_pow) - 1]
    #     elif q_pow < 0 and imag:
    #         return self.neg_inner_product_imag[-(q_pow) - 1]

    # def _calculate_inner_product(self):
    #     r"""Calculate the inner product according to the access.
    #
    #     If the access is "sparse", calculate the inner product using the sparce matrix estimator;
    #     If the access is "true", calculate the inner product using the matrix multiplication estimator;
    #     If the access is "sample", calculate the inner product using sampling and querying estimator;
    #     Else, calculate the inner product using the Hadamard test with backends provided by Qiskit;
    #     """
    #     if self.access == "sparse":
    #         if not isinstance(self.b, tuple):
    #             raise NotImplementedError("sparse mode is used with input Tuple[Dict[idx, value], size]")
    #         dict_b, size = self.b
    #         for i in range(self.power):
    #             self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = sparse_inner_product(dict_b, i + 1,
    #                                                                                                   size)
    #             self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = sparse_inner_product(dict_b, -(i + 1),
    #                                                                                                   size)
    #     elif self.access == "true" or self.access == "sample":
    #         if isinstance(self.b, np.ndarray):
    #             vec_b = self.b
    #         else:
    #             if isinstance(self.b, QuantumCircuit):
    #                 sim = Aer.get_backend('unitary_simulator')
    #                 job = execute(self.b, sim)
    #                 result = job.result()
    #                 mat = result.get_unitary(self.b, decimals=16)
    #                 vec_b = np.transpose(mat)[0]
    #         if self.access == "true":
    #             for i in range(self.power):
    #                 self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = true_inner_product(vec_b, i + 1)
    #                 self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = true_inner_product(vec_b, -(i + 1))
    #         elif self.access == "sample":
    #             for i in range(self.power):
    #                 self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = sample_inner_product(vec_b, i + 1,
    #                                                                                                       self.shots)
    #                 self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = sample_inner_product(vec_b,
    #                                                                                                       -(i + 1),
    #                                                                                                       self.shots)
    #     else:
    #         if isinstance(self.b, np.ndarray):
    #             width = int(np.log2(self.b.size))
    #             q_b = QuantumRegister(width, 'q')
    #             q_b_cir = QuantumCircuit(q_b)
    #             U_b = q_b_cir.prepare_state(state=Statevector(self.b)).instructions[0]
    #         else:
    #             U_b = self.b.to_gate()
    #             width = U_b.num_qubits
    #         promise_queue = []
    #         pr = []
    #         pi = []
    #         nr = []
    #         ni = []
    #         for i in range(self.power):
    #             pos_real = quantum_inner_product_promise(U_b, width, self.backend, i + 1, shots=self.shots, imag=False)
    #             pos_imag = quantum_inner_product_promise(U_b, width, self.backend, i + 1, shots=self.shots, imag=True)
    #             neg_real = quantum_inner_product_promise(U_b, width, self.backend, -(i + 1), shots=self.shots,
    #                                                      imag=False)
    #             neg_imag = quantum_inner_product_promise(U_b, width, self.backend, -(i + 1), shots=self.shots,
    #                                                      imag=True)
    #             promise_queue.append(pos_real)
    #             promise_queue.append(pos_imag)
    #             promise_queue.append(neg_real)
    #             promise_queue.append(neg_imag)
    #             pr.append(pos_real)
    #             pi.append(pos_imag)
    #             nr.append(neg_real)
    #             ni.append(neg_imag)
    #
    #         start = datetime.now()
    #         logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING,
    #                             handlers=[logging.FileHandler(f"queue_{start.strftime('%Y%m%d%H%M%S')}.log"),
    #                                       logging.StreamHandler()])
    #         logging.warning(f"access: {self.access}, shots: {self.shots}, power:{self.power}")
    #         time.sleep(self.power * 0.1)
    #         counter = len(promise_queue)
    #         while len(promise_queue) > 0:
    #             job = promise_queue.pop()
    #             status = job.status()
    #             counter -= 1
    #             if status == JobStatus.ERROR:
    #                 raise RuntimeError("Job failed.")
    #             elif status == JobStatus.CANCELLED:
    #                 raise RuntimeError("Job cancelled.")
    #             elif status == JobStatus.DONE:
    #                 logging.warning(f'Remaining jobs:{len(promise_queue)}')
    #                 counter = len(promise_queue)
    #             else:
    #                 promise_queue.append(job)
    #                 if counter == 0:
    #                     counter = len(promise_queue)
    #                     logging.warning('Waiting time: {:.2f} hours'.format((datetime.now() - start).seconds / 3600.0))
    #                     time.sleep(60 * 15)
    #         logging.warning('Queue cleared; total time: {:.2f} hours'.format((datetime.now() - start).seconds / 3600.0))
    #         for i in range(self.power):
    #             self.pos_inner_product_real[i] = eval_promise(pr[i])
    #             self.pos_inner_product_imag[i] = -eval_promise(pi[i])
    #             self.neg_inner_product_real[i] = eval_promise(nr[i])
    #             self.neg_inner_product_imag[i] = -eval_promise(ni[i])




########################################################################################################################


# if not {which_form}.issubset(['Pauli', 'Haar']):
#     raise ArgumentError(f"Invalid form: ({which_form})!\n"
#                         "Only 'Pauli' and 'Haar' are supported as the "
#                         "forms to generate the matrix. If you want to customize "
#                         "it, please input the matrix with parameter 'given_matrix'.",
#                         ModuleErrorCode,
#                         FileErrorCode, 1)

# if which_form == 'Pauli':
#     # Coefficients are sampled in [-2, 2] with uniform distribution
#     # self.__coeff = [(random.rand() * 4 - 2) for _ in range(self.__term_number)]
#     # self.__coeff = [1 for _ in range(int(self.__term_number / 2))] + [-1 for _ in range(int(self.__term_number / 2))]
#     # if len(self.__coeff) != self.__term_number:
#     #     self.__coeff += [random.choice([-1, 1])]
#     self.__coeff = [(random.rand() * 2 - 1) for _ in range(self.__term_number)]
#     # self.__coeff = [(random.rand() * 2) for _ in range(self.__term_number)]
#     # self.__coeff = random.normal(0, 2, self.__term_number)
