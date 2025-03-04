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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from transpiler.transpile import transpile_circuit, get_noisy_counts

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



def __run_circuit(qc, shots, **kwargs):
    transpile_kwargs = {i: kwargs[i] for i in ['device', 'optimization_level'] if i in kwargs.keys()}
    noisy_kwargs = {i: kwargs[i] for i in
                    ['noise_level_two_qubit', 'noise_level_one_qubit', 'readout_error']
                    if i in kwargs.keys()}
    cir_native = transpile_circuit(qc=qc, **transpile_kwargs)
    noisy_result = get_noisy_counts(qc=cir_native, shots=shots, **noisy_kwargs)
    p0 = 0
    p1 = 0
    if shots == 0:
        p0 = sum(noisy_result[:int(len(noisy_result) / 2)])
        p1 = sum(noisy_result[int(len(noisy_result) / 2):])
    else:
        noisy_result = {str(outcome): int(noisy_result[outcome]) for outcome in noisy_result.keys()}
        count0 = []
        count1 = []
        for outcome in noisy_result.keys():
            if outcome[0] == '0':
                count0.append(noisy_result[outcome])
            else:
                count1.append(noisy_result[outcome])
            if not count0:
                p0 = 0
                p1 = 1
            elif not count1:
                p0 = 1
                p1 = 0
            else:
                p0 = sum(count0) / shots
                p1 = sum(count1) / shots
    return p0 - p1


def Hadamard_test(n, U1, U2, Ub, shots, **kwargs):
    # build circuit
    cir_r = __build_circuit(n, U1, U2, Ub, alpha='r')
    exp_r = __run_circuit(cir_r, shots=int(shots / 2), **kwargs)
    cir_i = __build_circuit(n, U1, U2, Ub, alpha='i')
    exp_i = __run_circuit(cir_i, shots=int(shots / 2), **kwargs)
    expec = exp_r + exp_i * 1j
    return expec