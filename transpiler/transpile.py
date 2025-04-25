from bqskit import Circuit, compile
from bqskit.compiler.machine import MachineModel
from bqskit.ext import qiskit_to_bqskit
import re

import pennylane as qml
from qiskit import QuantumCircuit

from transpiler.bqskit_ionq_native_gates import GPIGate, GPI2Gate, PartialMSGate, ZZGate
from transpiler.qasm2_reader import load_qasm
import numpy as np
# Import required quantum computing libraries
from qiskit_aer import AerSimulator  # Qiskit's AerSimulator for running quantum circuits
from qiskit import transpile  # Used to optimize and compile circuits
from qiskit.transpiler import CouplingMap  # Defines the connectivity of a quantum processor
from qiskit.converters import circuit_to_dag, dag_to_circuit  # Convert between circuit and DAG representations
from collections import OrderedDict  # Used for ordered storage of quantum registers
from qiskit import QuantumCircuit

# Import required quantum computing libraries
from braket.circuits import Circuit

# ---------------------------------------------------------------------------
# **Quantum Hardware Mapping: Logical-to-Physical Qubit Layout**
# ---------------------------------------------------------------------------
# Mapping of logical qubits (software level) to physical qubits (hardware level)
# for IQM quantum processors with different numbers of qubits.
IQM_initial_layout = {
    '3': [9, 15, 11],
    '4': [9, 8, 14, 13],
    '5': [9, 8, 10, 14, 4],
    '6': [9, 8, 10, 14, 4, 13],
    '7': [9, 8, 10, 14, 4, 13, 15],
    '8': [9, 8, 10, 14, 4, 13, 15, 5],
    '9': [9, 8, 10, 14, 4, 13, 15, 5, 3],
    '10': [9, 8, 10, 14, 4, 13, 15, 5, 3, 11]
}

# **Quantum Coupling Map for IQM Device**
# This defines which qubits can interact directly on the hardware.
IQM_coupling_map = CouplingMap([
    (0, 1), (0, 3), (1, 4), (2, 3),
    (2, 7), (3, 4), (3, 8), (4, 5), (4, 9),
    (5, 6), (5, 10), (6, 11), (7, 8), (7, 12),
    (8, 9), (8, 13), (9, 10), (9, 14), (10, 11),
    (10, 15), (11, 16), (12, 13), (13, 14), (13, 17),
    (14, 15), (14, 18), (15, 19), (15, 16),
    (17, 18), (18, 19)
])


def __transpile_circuit_to_ionq(qc, device=None, optimization_level=2, synthesis_epsilon=1e-4,
                                max_synthesis_size=2):
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

    def transpile(circuit: Circuit):
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


def inverse_permutation(permutation):
    """
    Compute the inverse of a given permutation.

    Parameters:
        permutation (list): A list representing a permutation.

    Returns:
        list: The inverse permutation.
    """
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    return list(inverse)


def remove_idle_qwires(circ):
    """
    Remove idle quantum wires (qubits) from a given quantum circuit.

    Parameters:
        circ (QuantumCircuit): The quantum circuit to process.

    Returns:
        QuantumCircuit: A new circuit with idle qubits removed.
    """
    dag = circuit_to_dag(circ)  # Convert circuit to DAG representation

    # Identify and remove idle qubits
    idle_wires = list(dag.idle_wires())

    for w in idle_wires:
        dag._remove_idle_wire(w)
        if w in dag.qubits:
            dag.qubits.remove(w)

    # Clear quantum registers to reflect the updated circuit
    dag.qregs = OrderedDict()

    return dag_to_circuit(dag)  # Convert back to a circuit


# ---------------------------------------------------------------------------
# **Transpile Quantum Circuit for Simulation**
# ---------------------------------------------------------------------------

def __transpile_circuit_to_iqm_sim(qc: QuantumCircuit, optimization_level=2):
    """
    Transpile a given quantum circuit for simulation using AerSimulator.

    Parameters:
        qc (QuantumCircuit): The input quantum circuit.

    Returns:
        QuantumCircuit: The transpiled quantum circuit.
    """
    # Retrieve initial layout based on number of qubits
    initial_layout = IQM_initial_layout[str(qc.num_qubits)]

    # Initialize Qiskit's AerSimulator for running the circuit
    simulator = AerSimulator()

    # Perform Qiskit transpilation with optimization level
    qc = transpile(
        qc,
        simulator,
        optimization_level=optimization_level,
        basis_gates=['rx', 'ry', 'cz'],  # Allowed gates for hardware
        coupling_map=IQM_coupling_map,
        initial_layout=initial_layout
    )

    # Compute inverse layout for reversing logical-physical mapping
    inverse_layout = inverse_permutation(
        initial_layout + [i for i in range(20) if i not in initial_layout]
    )

    # Perform final transpilation with inverse layout and remove idle qubits
    qc_qiskit = transpile(qc, optimization_level=optimization_level, initial_layout=inverse_layout)

    qc_qiskit = remove_idle_qwires(qc_qiskit)

    return qc_qiskit  # Return optimized and cleaned-up circuit

def extract_indices(instruction_str):
    # Pattern to match the qubit index and clbit index
    pattern = r'Qubit\(.*?, (\d+)\).*?Clbit\(.*?, (\d+)\)'
    matches = re.search(pattern, instruction_str)
    if matches:
        return [int(matches.group(1)), int(matches.group(2))]
    return None

def __transpile_circuit_to_iqm(qc: QuantumCircuit, optimization_level=2) -> Circuit:

    qc.measure_all()

    # Retrieve initial layout based on number of qubits
    initial_layout = IQM_initial_layout[str(qc.num_qubits)]

    # Initialize Qiskit's AerSimulator for running the circuit
    simulator = AerSimulator()

    # Perform Qiskit transpilation with optimization level 3
    qc = transpile(
        qc,
        simulator,
        optimization_level=optimization_level,
        basis_gates=['rx', 'ry', 'cz'],  # Allowed gates for hardware
        coupling_map=IQM_coupling_map,
        initial_layout=initial_layout
    )

    measurement_correspondence=[extract_indices(str([qc.data[i]])) for i in range(len(qc.data)-qc.num_qubits,len(qc.data),1)]
    qc.remove_final_measurements()
    """
    Convert a Qiskit QuantumCircuit to an Amazon Braket Circuit.

    This function translates Qiskit gates into their corresponding Braket equivalents,
    specifically targeting **IQM quantum devices**.

    IQM devices support a **native gate set** that includes:
    - `prx(theta, alpha)`: Parameterized RX rotation (angle θ, axis shift α)
    - `cz`: Controlled-Z (CZ) gate

    Any additional gates are decomposed or mapped to this native set.

    Args:
        qc (QuantumCircuit): A Qiskit quantum circuit.

    Returns:
        Circuit: An Amazon Braket quantum circuit compatible with IQM devices.
    """

    braket_circuit = Circuit()  # Initialize an empty Braket circuit
    num_qubits = qc.num_qubits  # Get the number of qubits in the Qiskit circuit

    # Mapping of Qiskit gates to IQM's native Braket gates
    gate_map = {
        "h": lambda q, p: braket_circuit.h(q[0] + 1),  # Hadamard gate
        "x": lambda q, p: braket_circuit.x(q[0] + 1),  # Pauli-X gate
        "y": lambda q, p: braket_circuit.y(q[0] + 1),  # Pauli-Y gate
        "z": lambda q, p: braket_circuit.z(q[0] + 1),  # Pauli-Z gate

        # IQM-native RX (PRX) gate: prx(theta, alpha) with axis shift α
        "rx": lambda q, p: braket_circuit.prx(q[0] + 1, p[0], 0),  # RX(θ) with α=0
        "ry": lambda q, p: braket_circuit.prx(q[0] + 1, p[0], np.pi / 2),  # RY(θ) as RX(θ, π/2)
        "rz": lambda q, p: braket_circuit.rz(q[0] + 1, p[0]),  # RZ(θ) (directly supported)

        # Multi-qubit gates
        "cx": lambda q, p: braket_circuit.cnot(q[0] + 1, q[1] + 1),  # CNOT (CX) gate
        "cz": lambda q, p: braket_circuit.cz(q[0] + 1, q[1] + 1),  # **IQM-native CZ gate**
        "swap": lambda q, p: braket_circuit.swap(q[0] + 1, q[1] + 1),  # SWAP gate
        "ccx": lambda q, p: braket_circuit.ccnot(q[0] + 1, q[1] + 1, q[2] + 1),  # Toffoli (CCX) gate
    }

    # Iterate over Qiskit's circuit operations and convert them
    for instruction in qc.data:
        instr = instruction.operation  # Extract gate operation
        qargs = instruction.qubits  # Get the qubits involved in the operation
        qubits = [qc.find_bit(q).index for q in qargs]  # Extract qubit indices
        params = instr.params  # Get gate parameters (if any)

        if instr.name in gate_map:
            gate_map[instr.name](qubits, params)  # Apply the corresponding Braket gate
        else:
            raise ValueError(f"Unsupported gate: {instr.name}")  # Handle unsupported gates

    # Wrap the Braket circuit in a verbatim box and add measurement
    return measurement_correspondence, Circuit().add_verbatim_box(braket_circuit).measure(range(1, num_qubits + 1))


def transpile_circuit(qc, provider=None, device=None, optimization_level=2, synthesis_epsilon=1e-4,
                      max_synthesis_size=2):
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
    if provider is None:
        provider = 'ionq'
    if provider == 'ionq':
        qc_transpiled = __transpile_circuit_to_ionq(qc, device=device,
                                                    optimization_level=optimization_level,
                                                    synthesis_epsilon=synthesis_epsilon,
                                                    max_synthesis_size=max_synthesis_size)
        return qc_transpiled

    elif provider == 'iqm-sim':
        qc_transpiled = __transpile_circuit_to_iqm_sim(qc, optimization_level)
        return qc_transpiled

    elif provider == 'iqm':
        measurement_correspondence, qc_transpiled = __transpile_circuit_to_iqm(qc, optimization_level)
        return measurement_correspondence, qc_transpiled
    else:
        raise ValueError("Please provide a valid provider name: 'ionq' or 'iqm'.")


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
    print("Transpiled result", np.abs(qml.matrix(qml_circuit, wire_order=[0, 1, 2, 3])().T[0]) ** 2)  # check

    print("Noisy simulation result:", get_noisy_counts(qc, 0.00, 0.000, 0.0))
