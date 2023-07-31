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
    Access to the Qibo Simulator and the Quantum Hardware with noise.

    We self-developed the noisy model based on the fidelity of IonQ Aria-1.
    1Q   Fidelity    98.100% (avg)
    2Q   Fidelity    98.170% (avg)
    SPAM Error       99.400% (avg)

    We self-developed the noisy model based on the fidelity of IonQ Harmony.
    1Q   Fidelity    99.580% (avg)
    2Q   Fidelity    96.320% (avg)
    SPAM Error       99.752% (avg)
"""
from numpy import sqrt, arcsin
from qibo.models import Circuit
from qibo.gates import X, H, I, Z, Y, S, M, RX
from numpy import linalg

def Hadamard_test(U, alpha=1, shots=1024):
    # Initialize settings
    # U = [[], [], ...] := [[column1], [column2], ...]
    width = len(U[0])
    ancilla = 1

    # Aria
    # # 1Q Fidelity
    # one_Q_fidelity = 0.981
    # # 2Q Fidelity
    # two_Q_fidelity = 0.9817
    #
    # one_Q_error_percentage = 1 - one_Q_fidelity
    # two_Q_error_percentage = 1 - two_Q_fidelity

    one_Q_error_percentage = 0.0006
    two_Q_error_percentage = 0.006

    rot_ang_1Q = 2 * arcsin(sqrt(one_Q_error_percentage))
    rot_ang_2Q = 2 * arcsin(sqrt(two_Q_error_percentage))

    # # Harmony
    # # 1Q Fidelity
    # one_Q_fidelity = 0.9958
    # # 2Q Fidelity
    # two_Q_fidelity = 0.9958
    #
    # one_Q_error_percentage = 1 - one_Q_fidelity
    # two_Q_error_percentage = 1 - two_Q_fidelity
    #
    # rot_ang_1Q = 2 * arcsin(sqrt(one_Q_error_percentage))
    # rot_ang_2Q = 2 * arcsin(sqrt(two_Q_error_percentage))

    # When all tests are done, quantum hardware will be taken into consideration
    # simulator_backend = provider.get_backend("ionq_qpu")

    # Ancilla qubit as q0, others are tagged from q1 to qn
    # QuantumCircuit(4, 3) means a QuantumCircuit with 4 qubits and 3 classical bits
    Hadamard_circuit = Circuit(ancilla + width)
    Hadamard_circuit.add(RX(0, rot_ang_1Q))
    Hadamard_circuit.add(RX(1, rot_ang_1Q))
    Hadamard_circuit.add(H(0))
    Hadamard_circuit.add(RX(0, rot_ang_1Q))

    if alpha == 1j:
        Hadamard_circuit.add(S(0))
        Hadamard_circuit.add(RX(0, rot_ang_1Q))


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
            Hadamard_circuit.add(RX(i + 1, rot_ang_2Q).controlled_by(0))
    Hadamard_circuit.add(H(0))
    Hadamard_circuit.add(RX(0, rot_ang_1Q))

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
        print("The sampling result is:", result.frequencies())
        # p0 = 0.5
        # p1 = 0.5
        p0 = result.frequencies()['0'] / shots
        print("The sampling probability of getting 0 is:", p0)
        p1 = result.frequencies()['1'] / shots
        print("The sampling probability of getting 1 is:", p1)

        if p0 < 0.2:
            p0 = 0
            p1 = 1
            print("p0 < 0.2: set p0 = 0 and p1 = 1.")
            print("Expectation value is -1.")
        else:
            if p1 < 0.2:
                p0 = 1
                p1 = 0
                print("p0 > 0.8: set p0 = 1 and p1 = 0.")
                print("Expectation value is 1.")
            else:
                p0 = 0.5
                p1 = 0.5
                print("0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.")
                print("Expectation value is 0.")
        print()
    real_exp = p0 - p1
    return real_exp
