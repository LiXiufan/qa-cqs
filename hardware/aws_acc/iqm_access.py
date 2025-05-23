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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from braket.aws import AwsDevice

from qiskit import transpile as transpile_qiskit
from transpiler.transpile import transpile_circuit

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
    # For IQM, no measurement operation is developed at this stage.
    # cir.measure(anc[0], cr[0])
    backend = AerSimulator()
    # Transpile for optimization
    cir = transpile_qiskit(cir, backend, optimization_level=2)
    return cir


def __run_circuit(qc, shots, **kwargs):
    transpile_kwargs = {i: kwargs[i] for i in ['device', 'optimization_level'] if i in kwargs.keys()}

    # Transpile for iqm native circuits
    measurement_correspondence, qc_transpiled = transpile_circuit(qc=qc, provider='iqm', **transpile_kwargs)
    device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
    task = device.run(qc_transpiled, shots=shots, disable_qubit_rewiring=True)
    return measurement_correspondence, task.id


def Hadamard_test(n, U1, U2, Ub, shots, **kwargs):
    # build circuit
    cir_r = __build_circuit(n, U1, U2, Ub, alpha='r')
    cir_r_meas, cir_r_id = __run_circuit(cir_r, shots=int(shots / 2), **kwargs)
    cir_i = __build_circuit(n, U1, U2, Ub, alpha='i')
    cir_i_meas, cir_i_id = __run_circuit(cir_i, shots=int(shots / 2), **kwargs)
    cir_hadamard_test_id = ((cir_r_id, cir_i_id), (cir_r_meas, cir_i_meas))
    return cir_hadamard_test_id




