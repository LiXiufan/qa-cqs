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

from qiskit_braket_provider import AWSBraketProvider

BRAKET_DEVICE = 'SV1'
# BRAKET_DEVICE = 'Lucy'
# BRAKET_DEVICE = 'Aquila'
# BRAKET_DEVICE = 'Harmony'
# BRAKET_DEVICE = 'Aria 1'

# Hadamard test
def Hadamard_test(n, U1, U2, alpha='r', device='SV1', shots=1024):
    r"""
    Hadamard test to estimate <v1|v2> given two unitaries U1 and U2.

    Args:
        n (int): qubit number
        U1 (QuantumCircuit): unitary of left vector that U1|0>=|v1>
        U2 (QuantumCircuit): unitary of right vector that U2|0>=|v2>
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
    cir.append(U1.to_gate().control(ctrl_state='0'), [anc[0], *qr])
    cir.append(U2.to_gate().control(ctrl_state='1'), [anc[0], *qr])
    if alpha == 'i':
        cir.sdg(anc[0])
    cir.h(anc[0])
    cir.measure(anc[0], cr[0])

    # Transpile for simulator
    provider = AWSBraketProvider()
    backend = provider.get_backend(device)
    result = transpile(cir, backend=backend)

    job = backend.run(result, shots=shots)
    return job.job_id()

