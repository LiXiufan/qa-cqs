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
    Access to the Qibo Simulator and the Quantum Hardware.
"""

from qibo.models import Circuit
from qibo.gates import X, H, I, Z, Y, S, M
from numpy import linalg

def Hadamard_test(U, alpha=1, shots=1024):
    # Initialize settings
    # U = [[], [], ...] := [[column1], [column2], ...]
    width = len(U[0])
    ancilla = 1

    # When all tests are done, quantum hardware will be taken into consideration
    # simulator_backend = provider.get_backend("ionq_qpu")

    # Ancilla qubit as q0, others are tagged from q1 to qn
    # QuantumCircuit(4, 3) means a QuantumCircuit with 4 qubits and 3 classical bits
    Hadamard_circuit = Circuit(ancilla + width)
    Hadamard_circuit.add(H(0))
    if alpha == 1j:
        Hadamard_circuit.add(S(0))
    for layer in U:
        for i, gate in enumerate(layer):
            if gate == 'X':
                Hadamard_circuit.add(X(i + 1).controlled_by(0))
            elif gate == 'Y':
                Hadamard_circuit.add(Y(i + 1).controlled_by(0))
            elif gate == 'Z':
                Hadamard_circuit.add(Z(i + 1).controlled_by(0))
            elif gate == 'I':
                Hadamard_circuit.add(I(i + 1).controlled_by(0))
            elif gate == 'H':
                Hadamard_circuit.add(H(i + 1).controlled_by(0))
    Hadamard_circuit.add(H(0))
    Hadamard_circuit.add(M(0))
    result = Hadamard_circuit(nshots=shots)
    # print(result.frequencies()['0'])
    # print(result.frequencies()['1'])

    # Print the counts
    # print(job.get_counts())

    # Use sampling outcomes to approximate the probabilities and the real part of the expectation value
    if '0' not in result.frequencies().keys():
        p0 = 0
        p1 = 1
    elif '1' not in result.frequencies().keys():
        p0 = 1
        p1 = 0
    else:
        # p0 = 0.5
        # p1 = 0.5
        p0 = result.frequencies()['0'] / shots
        p1 = result.frequencies()['1'] / shots
    real_exp = p0 - p1
    return real_exp
