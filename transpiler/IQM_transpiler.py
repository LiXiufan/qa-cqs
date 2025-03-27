from __future__ import annotations  # Enables forward declarations (useful for type hints)
import re
# Import required quantum computing libraries
from qiskit_aer import AerSimulator  # Qiskit's AerSimulator for running quantum circuits
from qiskit import transpile  # Used to optimize and compile circuits
from qiskit.circuit.random import random_circuit  # Generates random quantum circuits
from qiskit.transpiler import CouplingMap  # Defines the connectivity of a quantum processor
from qiskit.converters import circuit_to_dag, dag_to_circuit  # Convert between circuit and DAG representations
from collections import OrderedDict  # Used for ordered storage of quantum registers
from qiskit import QuantumCircuit
from braket.circuits import Circuit
import qiskit
import numpy as np
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
    '7': [9, 8, 10, 14, 4, 13, 15]
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

# ---------------------------------------------------------------------------
# **Utility Functions**
# ---------------------------------------------------------------------------

def build_permutation_IQM(n, indices):
    """
    Builds a permutation π ∈ S_n such that elements at the given `indices`
    will be moved to positions 0, 1, 2, ..., respectively.

    Parameters:
        n (int): size of the permutation
        indices (list of int): list of original indices to be mapped to [0, 1, 2, ...]

    Returns:
        list: the resulting permutation of length n
    """
    permutation = [None] * n
    used_values = set()

    # Step 1: Assign specified positions
    for target_pos, original_index in enumerate(indices):
        permutation[original_index] = target_pos
        used_values.add(target_pos)

    # Step 2: Fill in remaining values without repetition
    free_values = iter([x for x in range(n) if x not in used_values])
    for i in range(n):
        if permutation[i] is None:
            permutation[i] = next(free_values)

    return permutation

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
        dag.qubits.remove(w)

    # Clear quantum registers to reflect the updated circuit
    dag.qregs = OrderedDict()

    return dag_to_circuit(dag)  # Convert back to a circuit

# ---------------------------------------------------------------------------
# **Transpile Quantum Circuit for Simulation**
# ---------------------------------------------------------------------------

def transpile_simulation(qiskit_qc):
    """
    Transpile a given quantum circuit for simulation using AerSimulator.

    Parameters:
        qiskit_qc (QuantumCircuit): The input quantum circuit.

    Returns:
        QuantumCircuit: The transpiled quantum circuit.
    """
    # Retrieve initial layout based on number of qubits
    initial_layout = IQM_initial_layout[str(qiskit_qc.num_qubits)]

    # Initialize Qiskit's AerSimulator for running the circuit
    simulator = AerSimulator()

    # Perform Qiskit transpilation with optimization level 3
    qc = transpile(
        qiskit_qc,
        simulator,
        optimization_level=3,
        basis_gates=['rx', 'ry', 'cz'],  # Allowed gates for hardware
        coupling_map=IQM_coupling_map,
        initial_layout=initial_layout
    )

    # Compute inverse layout for reversing logical-physical mapping
    inverse_layout = build_permutation_IQM(20,initial_layout)


    # Perform final transpilation with inverse layout and remove idle qubits
    qc_qiskit = transpile(qc, optimization_level=0, initial_layout=inverse_layout)

    qc_qiskit = remove_idle_qwires(qc_qiskit)

    return qc_qiskit  # Return optimized and cleaned-up circuit

def extract_indices(instruction_str):
    # Pattern to match the qubit index and clbit index
    pattern = r'Qubit\(.*?, (\d+)\).*?Clbit\(.*?, (\d+)\)'
    matches = re.search(pattern, instruction_str)
    if matches:
        return [int(matches.group(1)), int(matches.group(2))]
    return None

def transpile_to_IQM_braket(qiskit_qc: QuantumCircuit) -> Circuit:

    qiskit_qc.measure_all()

    # Retrieve initial layout based on number of qubits
    initial_layout = IQM_initial_layout[str(qiskit_qc.num_qubits)]

    # Initialize Qiskit's AerSimulator for running the circuit
    simulator = AerSimulator()

    # Perform Qiskit transpilation with optimization level 3
    qc = transpile(
        qiskit_qc,
        simulator,
        optimization_level=3,
        basis_gates=['rx', 'ry', 'cz'],  # Allowed gates for hardware
        coupling_map=IQM_coupling_map,
        initial_layout=initial_layout
    )

    measurement_correspondence=[extract_indices(str([qc.data[i]])) for i in range(len(qc.data)-qiskit_qc.num_qubits,len(qc.data),1)]
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
    return measurement_correspondence,Circuit().add_verbatim_box(braket_circuit).measure(range(1, num_qubits + 1))


# ---------------------------------------------------------------------------
# **Main Execution: Generate & Transpile a Random Quantum Circuit**
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    def print_sorted_binary_dict(data):
        """
        Prints dictionary entries in two columns:
        - Left: sorted by binary keys (ascending order)
        - Right: sorted by values (descending order)
        """
        sorted_by_key = sorted(data.items(), key=lambda item: int(item[0], 2))
        sorted_by_value = sorted(data.items(), key=lambda item: item[1], reverse=True)

        key_width = max(len(key) for key in data)

        print(f"{'By Key':<{key_width+8}}{'':4}{'By Value'}")
        print("-" * (key_width + 8) + "    " + "-" * (key_width + 8))

        for left, right in zip(sorted_by_key, sorted_by_value):
            l_key, l_val = left
            r_key, r_val = right
            print(f"{l_key}: {l_val:<5}    {r_key}: {r_val}")




    n = 6
    # Generate a random 5-qubit quantum circuit with a depth of 10
    qc = random_circuit(n, max_operands=2, depth=10, measure=False)
    # qc=QuantumCircuit(5)
    # qc.rx(1,0)
    # qc.measure_all()

    measurement_correspondence,IQM_braket_circuit = transpile_to_IQM_braket(qc)
    print(measurement_correspondence)
    # # transpile_to_IQM_braket(qc)
    #
    #
    # simulator = AerSimulator()
    # qc2=qc.copy()
    # qc2=transpile(qc2, simulator)
    # result1 = simulator.run(qc2, shots=10000).result()
    # print_sorted_binary_dict(result1.get_counts())
    #
    # print()
    # result2 = simulator.run(transpiled_qc_sim, shots=10000).result()
    # print_sorted_binary_dict(result2.get_counts())


