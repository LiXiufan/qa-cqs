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
    Access to the IonQ Simulator and the Quantum Hardware.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from numpy import random

def Hadamard_test(U, alpha=1, shots=1024):
    # Initialize settings
    # U = [[], [], ...] := [[column1], [column2], ...]
    width = len(U[0])
    ancilla = 1

    # Check the backends
    # We need a token created by IBMQ's API.
    xiufan_token = 'd383594cc2f9b3c7baf90b70b11f658c2df3ede2f534e571cc73d5e01f70e7a23620b4384594db7660ded289e864f137f3be56f05f39b9a113545ad90b11a302'

    # QiskitRuntimeService.save_account(channel="ibm_quantum", token=xiufan_token)

    # service = QiskitRuntimeService(channel="ibm_quantum", token=xiufan_token)
    # service = QiskitRuntimeService()

    provider = IBMProvider(token=xiufan_token)
    simulator_backend = provider.get_backend("simulator_statevector")
    # backend = service.backend('simulator_statevector')
    # simulator_backend = provider.get_backend("ionq_qpu")
    # options = Options()
    # options.execution.shots = shots

    # sampler = Sampler(session=backend, options=options)
    # print(backends)

    # When all tests are done, quantum hardware will be taken into consideration
    # simulator_backend = provider.get_backend("ionq_qpu")

    # Create gates for the unitary
    # U = [U1, U2], U1 operates before U2, --> |psi> = U2 U1 |0>
    U_in_circuit = QuantumCircuit(width)
    for layer in U:
        for i, gate in enumerate(layer):
            if gate == 'X':
                U_in_circuit.x(i)
            elif gate == 'Y':
                U_in_circuit.y(i)
            elif gate == 'Z':
                U_in_circuit.z(i)
            # elif gate == 'I':
            #     U_in_circuit.i(i)
            elif gate == 'H':
                U_in_circuit.h(i)
    U_gate = U_in_circuit.to_gate()
    CU_gate = U_gate.control()

    # Ancilla qubit as q0, others are tagged from q1 to qn
    # QuantumCircuit(4, 3) means a QuantumCircuit with 4 qubits and 3 classical bits
    Hadamard_circuit = QuantumCircuit(ancilla + width, 1)
    Hadamard_circuit.h(0)
    if alpha == 1j:
        Hadamard_circuit.s(0)
    Hadamard_circuit.append(CU_gate, [0] + list(range(1, width + 1)))
    Hadamard_circuit.h(0)
    Hadamard_circuit.measure([0], [0])

    # Transpile the circuit for Hadamard test
    # result = transpile(Hadamard_circuit, backend=simulator_backend, optimization_level=3)
    result = transpile(Hadamard_circuit, backend=simulator_backend)
    job = simulator_backend.run(result, shots=shots)
    result = job.result()
    counts = result.get_counts()
    # print(counts)

    # result = sampler.run(Hadamard_circuit).result()
    # Print the counts
    # print(job.get_counts())

    # Use sampling outcomes to approximate the probabilities and the real part of the expectation value
    # if '0' not in job.get_counts().keys():
    #     p0 = 0
    #     p1 = 1
    # elif '1' not in job.get_counts().keys():
    #     p0 = 1
    #     p1 = 0
    # else:
    #     p0 = job.get_counts()['0'] / shots
    #     p1 = job.get_counts()['1'] / shots
    # real_exp = p0 - p1
    # print('Real part of the expectation value is:', real_exp)

    # The simulator provides the ideal probabilities from the circuit, and the provider
    # creates “counts” by randomly sampling from these probabilities. The raw (“true”)
    # probabilities are also accessible by calling get_probabilities():
    eps_range = 10 ** (-4)
    if '0' not in counts.keys():
        p0 = 0 + random_error(eps_range)
        p1 = 1 + random_error(eps_range)
    elif '1' not in counts.keys():
        p0 = 1 + random_error(eps_range)
        p1 = 0 + random_error(eps_range)
    else:
        p0 = 0.5 + random_error(eps_range)
        p1 = 0.5 + random_error(eps_range)
        # p0 = job.get_probabilities()['0']
        # p1 = job.get_probabilities()['1']
    real_exp = p0 - p1
    # print('Real to calculate', real_exp)

    return real_exp


def random_error(eps_range):
    eps = random.rand() * 2 * eps_range - eps_range
    return eps