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
    This is the matrix calculation for Hadamard test.
"""

from numpy import kron, conj, transpose, real, imag
from cqs_module.verifier import get_unitary, zero_state
def Hadmard_test(U, alpha=1):
    U_mat = get_unitary(U)
    width = len(U[0])
    zeros = zero_state()
    if width > 1:
        for j in range(width - 1):
            zeros = kron(zeros, zero_state())

    ideal = (conj(transpose(zeros)) @ U_mat @ zeros).item()
    if alpha == 1:
        return real(ideal)
    elif alpha == 1j:
        return imag(ideal)
    else:
        raise ValueError("The alpha should be either 1 or 1j.")