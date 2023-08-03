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
    file1 = open(file_name, "a")
    file1.writelines(["The unitary for estimation is:", str(U), '\n'])
    width = len(U[0])
    ancilla = 1

    # When all tests are done, quantum hardware will be taken into consideration
    # simulator_backend = provider.get_backend("ionq_qpu")

    # Ancilla qubit as q0, others are tagged from q1 to qn
    # QuantumCircuit(4, 3) means a QuantumCircuit with 4 qubits and 3 classical bits
    circuit = Circuit(ancilla + width)
    circuit.add(H(0))
    if alpha == 1j:
        circuit.add(S(0))
    for layer in U:
        for i, gate in enumerate(layer):
            if gate == 'X':
                circuit.add(X(i + 1).controlled_by(0))
            elif gate == 'Y':
                circuit.add(Y(i + 1).controlled_by(0))
            elif gate == 'Z':
                circuit.add(Z(i + 1).controlled_by(0))
            elif gate == 'I':
                circuit.add(I(i + 1).controlled_by(0))
            elif gate == 'H':
                circuit.add(H(i + 1).controlled_by(0))
    circuit.add(H(0))
    circuit.add(M(0))
    result = circuit(nshots=shots)
    # print(result.frequencies()['0'])
    # print(result.frequencies()['1'])

    # Print the counts
    # print(job.get_counts())

    # Use sampling outcomes to approximate the probabilities and the real part of the expectation value
    file1.writelines(["The sampling result is:", str(result.frequencies()), '\n'])
    # print("The sampling result is:", result.frequencies())
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
        # print("The sampling probability of getting 0 is:", p0)
        # print("The sampling probability of getting 1 is:", p1)
        file1.writelines(["The sampling probability of getting 0 is: p0 =", str(p0), '\n'])
        file1.writelines(["The sampling probability of getting 1 is: p1 =", str(p1), '\n'])

        if p0 < 0.2:
            p0 = 0
            p1 = 1
            # print("p0 < 0.2: set p0 = 0 and p1 = 1.")
            # print("Expectation value is -1.")
            file1.writelines(["p0 < 0.2: set p0 = 0 and p1 = 1.", '\n'])
            file1.writelines(["Expectation value is -1.", '\n'])
        else:
            if p1 < 0.2:
                p0 = 1
                p1 = 0
                # print("p0 > 0.8: set p0 = 1 and p1 = 0.")
                # print("Expectation value is 1.")
                file1.writelines(["p0 > 0.8: set p0 = 1 and p1 = 0.", '\n'])
                file1.writelines(["Expectation value is 1.", '\n'])
            else:
                p0 = 0.5
                p1 = 0.5
                # print("0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.")
                # print("Expectation value is 0.")
                file1.writelines(["0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.", '\n'])
                file1.writelines(["Expectation value is 0.", '\n'])
        # print()
    file1.close()
    real_exp = p0 - p1
    return real_exp
