from __future__ import annotations
from qiskit import QuantumCircuit, transpile

from qiskit_ionq import IonQProvider

from transpiler.bqskit_ionq_native_gates import GPIGate, GPI2Gate, PartialMSGate
from transpiler.transpile import transpile_circuit, get_noisy_counts

IONQ_GATES = {PartialMSGate(), GPIGate(), GPI2Gate()}
OPT_LEVEL = 2

if __name__ == "__main__":
    # ===========================
    # TEST: Transpile a Qiskit circuit
    # ===========================

    # Create a Qiskit quantum circuit
    qc_qiskit = QuantumCircuit(4)
    qc_qiskit.h(0)  # Hadamard on qubit 0
    qc_qiskit.rzz(1, 0, 1)  # RZZ gate with parameter 1
    qc_qiskit.ry(3, 1)  # RY gate with parameter 3 on qubit 1
    qc_qiskit.ry(4, 2)  # RY gate with parameter 3 on qubit 1
    qc_qiskit.ry(5, 3)  # RY gate with parameter 3 on qubit 1
    qc_qiskit.rxx(2.14584545, 0, 1)  # RXX gate with parameter
    qc_qiskit.rxx(2.14584545, 2, 3)  # RXX gate with parameter
    qc_qiskit.rxx(2.14584545, 1, 2)  # RXX gate with parameter
    qc_qiskit.rxx(2.14584545, 0, 3)  # RXX gate with parameter
    qc_qiskit.rzz(1.1561, 1, 0)  # RZZ gate with parameter
    qc_qiskit.rx(1.1561, 1)  # RX gate with parameter

    # Print the transpiled circuit
    print("TEST Qiskit Circuit:")
    print(qc_qiskit)

    # Transpile for iqm native circuits
    transpiled_qc = transpile_circuit(qc_qiskit, provider='iqm-sim', optimization_level=OPT_LEVEL)

    # Print the transpiled circuit
    print("Transpiled Qiskit Circuit:")
    print(transpiled_qc)

    # GET NOISEless probabilities
    print(get_noisy_counts(transpiled_qc, 0, 0, 0))
    print(get_noisy_counts(transpiled_qc, 0, 0, 0))
    print(get_noisy_counts(transpiled_qc, 0, 0, 0))
    print("lalalalal")

    # GET NOISY probabilities without the presence of shot noise
    print(get_noisy_counts(transpiled_qc, 0, 0.02, 0, 0))


