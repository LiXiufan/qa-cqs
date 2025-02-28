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
    This is the matrix calculation using qiskit aer simulator.
"""

from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

# Hadamard test
def __build_circuit(n, U1, U2, Ub, alpha='r'):
    r"""
    Hadamard test to estimate <b| U1^{\dagger} U2 |b> given two unitaries U1, U2, and the state preparation circuit Ub.

    Args:
        n (int): qubit number
        U1 (QuantumCircuit): unitary of left vector that U1|0>=|v1>
        U2 (QuantumCircuit): unitary of right vector that U2|0>=|v2>
        U2 (QuantumCircuit): unitary of state preparation that Ub|0>=|b>
        alpha (str): 'r' or 'i', real or imaginary

    Returns:
        QuantumCircuit: the circuit of Hadamard test
    """
    if alpha not in ['r', 'i']:
        raise ValueError("Please specify the real part or the imaginary part using 'r' or 'i'.")
    anc = QuantumRegister(1, 'ancilla')
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(1, 'c')
    cir = QuantumCircuit(anc, qr, cr)
    cir.h(anc[0])
    cir.append(Ub.to_gate(), [*qr])
    cir.append(U1.to_gate().control(ctrl_state='0'), [anc[0], *qr])
    cir.append(U2.to_gate().control(ctrl_state='1'), [anc[0], *qr])
    if alpha == 'i':
        cir.sdg(anc[0])
    cir.h(anc[0])
    cir.measure(anc[0], cr[0])
    return cir

def __run_circuit(cir, shots=1024):
    # Transpile for simulator
    simulator = AerSimulator()
    cir = transpile(cir, simulator)
    if shots == 0:
        # Run and get probabilities
        cir.remove_final_measurements()
        state_vec = Statevector(cir)
        prob_zero_qubit = state_vec.probabilities([0])
        p0 = prob_zero_qubit[0]
        p1 = prob_zero_qubit[1]
    else:
        # Run and get counts
        result = simulator.run(cir, shots=shots).result()
        counts = result.get_counts(0)
        if '0' not in counts.keys():
            p0 = 0
            p1 = 1
        elif '1' not in counts.keys():
            p0 = 1
            p1 = 0
        else:
            p0 = counts['0'] / shots
            p1 = counts['1'] / shots
    return p0 - p1

def Hadamard_test(n, U1, U2, Ub, shots=1024, noise_level=None):
    # build circuit
    cir_r = __build_circuit(n, U1, U2, Ub, alpha='r')
    cir_i = __build_circuit(n, U1, U2, Ub, alpha='i')








    exp_r = __run_circuit(, shots=int(shots / 2))
    exp_i = __run_circuit(, shots=int(shots / 2))
    expec = exp_r + exp_i * 1j
    return expec


