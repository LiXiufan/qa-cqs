from bqskit import Circuit, compile
from bqskit.compiler.machine import MachineModel
from bqskit.ext import qiskit_to_bqskit

import pennylane as qml
from qiskit import QuantumCircuit

from transpiler.bqskit_ionq_native_gates import GPIGate, GPI2Gate, PartialMSGate, ZZGate
from transpiler.qasm2_reader import load_qasm
import numpy as np


def transpile_circuit(qc, device=None, optimization_level=2, synthesis_epsilon=1e-4, max_synthesis_size=2):
    """
    Transpiles a Qiskit quantum circuit to use only MS (Mølmer–Sørensen) gates
    using BQSKit, and returns the transpiled Qiskit circuit.

    Args:
        qc (QuantumCircuit): The input Qiskit quantum circuit.
        device (str): The target device from IONQ devices with corresponding IONQ_gate_set.
            option 1: "Aria" {PartialMSGate(), GPIGate(), GPI2Gate()}
            option 2: "Forte" {ZZGate(), GPIGate(), GPI2Gate()}
    Returns:
        QuantumCircuit: The transpiled Qiskit quantum circuit with only MS gates.
    """
    if device is None:
        device = "Aria"
    # Convert the Qiskit circuit to a BQSKit circuit
    if device == "Aria":
        gate_set = {PartialMSGate(), GPIGate(), GPI2Gate()}
    elif device == "Forte":
        gate_set = {ZZGate(), GPIGate(), GPI2Gate()}
    bqskit_qc = qiskit_to_bqskit(qc)

    def transpile(circuit: Circuit) -> Circuit:
        """
        Transpiles a given BQSKit circuit to use only MS gates.

        Args:
            circuit (Circuit): The input BQSKit quantum circuit.

        Returns:
            Circuit: The transpiled circuit with MS gates.
        """
        # Define the machine model with the target gate set
        model = MachineModel(circuit.num_qudits, gate_set=gate_set)

        # Compile the circuit with the given model at optimization level 2
        compiled_circuit = compile(circuit, model, optimization_level=optimization_level,
                                   synthesis_epsilon=synthesis_epsilon, max_synthesis_size=max_synthesis_size)

        return compiled_circuit

    # Transpile the BQSKit circuit to use only MS gates
    ms_circuit = transpile(bqskit_qc)

    # Save the transpiled circuit as a QASM file
    ms_circuit.save("circuit.qasm")

    # Load the QASM file back into a Qiskit circuit
    qc_transpiled = load_qasm("circuit.qasm")

    # Return the final Qiskit circuit
    return qc_transpiled


def get_noisy_counts(qc, shots=None, noise_level_two_qubit=None, noise_level_one_qubit=None, readout_error=None):
    shots_flag = True
    if shots is None:
        shots = 0
    if shots == 0:
        shots = 1024
        shots_flag = False
    if noise_level_one_qubit is None:
        noise_level_one_qubit = 0
    if noise_level_two_qubit is None:
        noise_level_two_qubit = 0
    if readout_error is None:
        readout_error = 0

    # Convert the Qiskit quantum circuit into a PennyLane circuit
    qml_circuit = qml.from_qiskit(qc)

    # Compute the adjusted noise level for single-qubit depolarizing channels in 2Q noise
    noise_level_two_qubit_single_channel = noise_level_two_qubit * 3 / 4

    # Get the number of qubits in the circuit
    number_of_qubits = qc.num_qubits

    # Define a QNode using PennyLane's mixed-state simulator (supports noise)
    @qml.qnode(qml.device("default.mixed", wires=number_of_qubits, shots=shots))
    def noisy_circuit():
        # Convert the PennyLane circuit into a tape (sequence of operations)
        tape = qml.transforms.make_tape(qml_circuit)()

        # Iterate through all operations in the circuit
        for op in tape.operations:
            gate = getattr(qml, op.name, None)  # Dynamically get the gate

            if gate:
                try:
                    gate(*op.parameters, wires=op.wires)  # Apply gate with parameters
                except TypeError:
                    raise TypeError(f"Unexpected parameters for gate {op.name}: {op.parameters}")
            else:
                raise ValueError(f"Unknown gate {op.name} encountered. Please check the circuit.")

            # Apply **1-qubit depolarizing noise** after every 1-qubit gate
            if len(op.wires) == 1:
                qml.DepolarizingChannel(noise_level_one_qubit, wires=op.wires[0])

            # Apply **2-qubit depolarizing noise** after two-qubit gates
            if len(op.wires) == 2:
                for qubit in op.wires:
                    qml.DepolarizingChannel(noise_level_two_qubit_single_channel, wires=qubit)

        # Apply **readout error (bit flip) before measurement**
        for qubit in range(number_of_qubits):
            qml.BitFlip(readout_error, wires=qubit)

        if shots_flag:
            # Return the outcome count given by number of shots
            return qml.counts(wires=range(number_of_qubits))
        else:
            # Return the probability distribution over computational basis states
            return qml.probs(wires=range(number_of_qubits))

    # Run the noisy circuit simulation and return the result
    noisy_result = noisy_circuit()
    return noisy_result


if __name__ == "__main__":
    qc = load_qasm("circuit.qasm")
    print(qc)
    qml_circuit = qml.from_qiskit(qc)
    print("Transpiled result", np.abs(qml.matrix(qml_circuit, wire_order=[0, 1,2,3])().T[0]) ** 2)  # check

    print("Noisy simulation result:", get_noisy_counts(qc, 0.00, 0.000, 0.0))
