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
    Access to the AWS Braket Simulator and the Quantum Hardware.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from braket.aws import AwsDevice
from qiskit_braket_provider import AWSBraketProvider
from braket.circuits import Circuit

from transpiler.transpile import transpile_circuit
from transpiler.qasm2_reader import from_qasm2_to_braket

BRAKET_DEVICE = 'SV1'
# BRAKET_DEVICE = 'Lucy'
# BRAKET_DEVICE = 'Aquila'
# BRAKET_DEVICE = 'Harmony'
# BRAKET_DEVICE = 'Aria 1'

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
    backend = AerSimulator()
    # Transpile for optimization
    cir = transpile(cir, backend, optimization_level=3)
    return cir


def __run_circuit(qc, shots, **kwargs):
    transpile_kwargs = {i: kwargs[i] for i in ['device', 'optimization_level'] if i in kwargs.keys()}

    # Transpile for ionq native circuits
    cir_native = transpile_circuit(qc=qc, **transpile_kwargs)
    circuit_qasm = from_qasm2_to_braket('circuit.qasm')

    # Try with Aria-1
    device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")
    task = device.run(circuit_qasm, shots=shots, disable_qubit_rewiring=True)
    return task.id


def Hadamard_test(n, U1, U2, Ub, shots, **kwargs):
    # build circuit
    cir_r = __build_circuit(n, U1, U2, Ub, alpha='r')
    cir_r_id = __run_circuit(cir_r, shots=int(shots / 2), **kwargs)
    cir_i = __build_circuit(n, U1, U2, Ub, alpha='i')
    cir_i_id = __run_circuit(cir_i, shots=int(shots / 2), **kwargs)
    cir_hadamard_test_id = [cir_r_id, cir_i_id]
    return cir_hadamard_test_id




