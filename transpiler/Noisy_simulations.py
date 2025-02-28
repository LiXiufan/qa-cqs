from __future__ import annotations
from bqskit import Circuit, compile
from bqskit.compiler.machine import MachineModel
from bqskit.ir.gates import RXXGate, RXGate, RZGate,RYGate,RZZGate,HGate,CNOTGate
from transpiler.BQSkit_custom_gate_definitions import VirtualZGate, GPIGate, GPI2Gate,PartialMSGate,FullMSGate
from transpiler.QASM2_reader import load_qasm
import pennylane as qml
from bqskit.ext import qiskit_to_bqskit
from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile

from qiskit_ionq import IonQProvider


IONQ_gate_set = {PartialMSGate(), GPIGate(), GPI2Gate()}
def transpiler(qiskit_qc, gate_set=IONQ_gate_set,optimization_level=2):
    """
    Transpiles a Qiskit quantum circuit to use only MS (Mølmer–Sørensen) gates
    using BQSKit, and returns the transpiled Qiskit circuit.

    Args:
        qiskit_qc (QuantumCircuit): The input Qiskit quantum circuit.
        gate_set (set, optional): The target gate set for transpilation.
                                  Defaults to IONQ_gate_set.

    Returns:
        QuantumCircuit: The transpiled Qiskit quantum circuit with only MS gates.
    """

    # Convert the Qiskit circuit to a BQSKit circuit
    bqskit_qc = qiskit_to_bqskit(qiskit_qc)

    def transpile_to_ms(circuit: Circuit) -> Circuit:
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
        compiled_circuit = compile(circuit, model, optimization_level=optimization_level)

        return compiled_circuit

    # Transpile the BQSKit circuit to use only MS gates
    ms_circuit = transpile_to_ms(bqskit_qc)

    # Save the transpiled circuit as a QASM file
    ms_circuit.save("circuit.qasm")

    # Load the QASM file back into a Qiskit circuit
    qiskit_circuit_transpiled = load_qasm("circuit.qasm")

    # Return the final Qiskit circuit
    return qiskit_circuit_transpiled



def get_noisy_counts(qiskit_qc, noise_level_two_qubit, noise_level_one_qubit, readout_error):
    # Convert the Qiskit quantum circuit into a PennyLane circuit
    qml_circuit = qml.from_qiskit(qiskit_qc)

    # Compute the adjusted noise level for single-qubit depolarizing channels in 2Q noise
    noise_level_two_qubit_single_channel = noise_level_two_qubit * 3 / 4

    # Get the number of qubits in the circuit
    number_of_qubits = qiskit_qc.num_qubits

    # Define a QNode using PennyLane's mixed-state simulator (supports noise)
    @qml.qnode(qml.device("default.mixed", wires=number_of_qubits))
    def noisy_circuit():
        # Convert the PennyLane circuit into a tape (sequence of operations)
        tape = qml.transforms.make_tape(qml_circuit)()

        # Iterate through all operations in the circuit
        for op in tape.operations:
            # Apply each gate as a unitary operation
            qml.QubitUnitary(op.parameters[0], wires=op.wires)

            # Apply **1-qubit depolarizing noise** after every 1-qubit gate
            if len(op.wires) == 1:
                qml.DepolarizingChannel(noise_level_one_qubit, wires=op.wires[0])

            # Apply **2-qubit depolarizing noise** after two-qubit (MS) gates
            if len(op.wires) == 2:
                for qubit in op.wires:
                    qml.DepolarizingChannel(noise_level_two_qubit_single_channel, wires=qubit)  # Adjusted probability

        # Apply **readout error (bit flip) before measurement**
        for qubit in range(number_of_qubits):
            qml.BitFlip(readout_error, wires=qubit)

        # Return the probability distribution over computational basis states
        return qml.probs(wires=range(number_of_qubits))

    # Run the noisy circuit simulation and return the result
    noisy_result = noisy_circuit()
    return noisy_result



if __name__=="__main__":

    # ===========================
    # TEST: Transpile a Qiskit circuit
    # ===========================

    # Create a Qiskit quantum circuit
    qc_qiskit = QuantumCircuit(2)
    qc_qiskit.h(0)  # Hadamard on qubit 0
    qc_qiskit.rzz(1, 0, 1)  # RZZ gate with parameter 1
    qc_qiskit.ry(3, 1)  # RY gate with parameter 3 on qubit 1
    qc_qiskit.rxx(2.14584545, 0, 1)  # RXX gate with parameter
    qc_qiskit.rzz(1.1561, 1, 0)  # RZZ gate with parameter
    qc_qiskit.rx(1.1561, 1)  # RX gate with parameter

    # Print the transpiled circuit
    print("TEST Qiskit Circuit:")
    print(qc_qiskit)

    # Transpile the Qiskit circuit to use MS gates
    transpiled_qc = transpiler(qc_qiskit)
    # GET NOISEless probabilities
    print(get_noisy_counts(transpiled_qc, 0,0,0))
    # Print the transpiled circuit
    print("Transpiled Qiskit Circuit:")
    print(transpiled_qc)
    #GET NOISY probabilities
    print(get_noisy_counts(transpiled_qc, 0.02,0,0))




    provider = IonQProvider()
    backend_native = provider.get_backend("simulator", gateset="native")
    transpiled_qc = transpile(qc_qiskit, backend=backend_native, optimization_level=3)

    # GET NOISEless probabilities
    print(get_noisy_counts(transpiled_qc, 0,0,0))

    # Print the transpiled circuit
    print("Transpiled Qiskit Circuit:")
    print(transpiled_qc)

    #GET NOISY probabilities
    print(get_noisy_counts(transpiled_qc, 0.02,0.001,0.1))
