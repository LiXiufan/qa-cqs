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

r"""
    Test the shot frugal method for the summation of different expectations obtained by Hadamard tests.
    H = \sum_{k=1}^{K} \beta_k U_k
    Goal: Estimate the <b|H|b> with a limited number of shots.

    Assume: all betas are in [-1, 1]
"""
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from random import random, choice
from numpy import linalg

def generate_case(n, L):
    d = n
    # random unitaries
    U = [random_circuit(n, d, measure=False) for _ in range(L)]
    U_b = random_circuit(n, d, measure=False)
    # random coefficients
    BETA = [random() * 2 - 1 for _ in range(L)]
    return U, U_b, BETA


# Hadamard test
def Hadamard_test(n, U1, U2, alpha='r'):
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
    return cir


def compute_shots(BETA, SHOTS):
    L = len(BETA)
    beta_abs = [abs(i) for i in BETA]
    B = sum(beta_abs)

    shots_allocated = {'uniform': [int(SHOTS / L) for _ in range(L)],
                       'weighted': [int((SHOTS / B) * beta_abs[j]) for j in range(L)],
                       'exact': [0 for _ in range(L)]}
    return shots_allocated


def run_circuit(cir, shots=1024):
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


def eval_exp(n, U1, U2, shots=1024):
    exp_r = run_circuit(Hadamard_test(n, U1, U2, alpha='r'), shots=int(shots / 2))
    exp_i = run_circuit(Hadamard_test(n, U1, U2, alpha='i'), shots=int(shots / 2))
    expec = exp_r + exp_i * 1j
    return expec



def main(NRANGE, SHOTS, SAMPLE, FILE):
    LABLES = ['uniform', 'weighted', 'exact']
    file1 = open(FILE, "a")
    for itr in range(SAMPLE):
        n = choice(NRANGE)
        L = choice(list(range(3, 2 * n)))
        U, U_b, BETA = generate_case(n, L)
        H_exp = {i: 0 for i in LABLES}
        shots_allocated = compute_shots(BETA, SHOTS)

        file1.writelines(["Itr:", str(itr), '\n'])
        file1.writelines(["L:", str(L), '\n'])
        file1.writelines(["Beta:", str(BETA), '\n'])
        file1.writelines(["UDS shots:", str(shots_allocated['uniform']), '\n'])
        file1.writelines(["WDS shots:", str(shots_allocated['weighted']), '\n'])

        for label in LABLES:
            for j in range(L):
                expec = eval_exp(n, U_b, U_b.compose(U[j]), shots=shots_allocated[label][j])
                H_exp[label] += BETA[j] * expec

        error_uniform = linalg.norm(H_exp['uniform'] - H_exp['exact']).item()
        error_weighted = linalg.norm(H_exp['weighted'] - H_exp['exact']).item()
        file1.writelines(["UDS error:", str(error_uniform), '\n'])
        file1.writelines(["WDS error:", str(error_weighted), '\n'])
        file1.writelines(['\n'])
    # return ErrorUniform, ErrorWeighted