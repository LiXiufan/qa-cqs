from qiskit import QuantumCircuit  # Import the QuantumCircuit class from Qiskit
from transpiler.qiskit_custom_gate_definitions import VirtualZGate,GPIGate,GPI2Gate,FullMSGate,PartialMSGate


# def load_qasm(file_path):
#     """
#     Load an OpenQASM 2.0 file and construct a Qiskit QuantumCircuit.
#
#     This function:
#     - Reads a QASM file line by line.
#     - Parses standard gates like `rx`, `ry`, `rz`, `cx`, etc.
#     - Replaces `GPI`, `GPI2`, and `virt_z` with their custom gate definitions.
#
#     Args:
#         file_path (str): Path to the QASM file.
#
#     Returns:
#         QuantumCircuit: A Qiskit circuit containing the parsed gates.
#     """
#
#     # Read the QASM file
#     with open(file_path, "r") as f:
#         lines = f.readlines()
#
#     # Determine the number of qubits from qreg
#     num_qubits = 0
#     for line in lines:
#         if "qreg q[" in line:
#             num_qubits = int(line.split("[")[1].split("]")[0])
#
#     # Initialize a QuantumCircuit
#     qc = QuantumCircuit(num_qubits)
#
#     # Parse QASM lines
#     for line in lines:
#         tokens = line.strip().split()
#         if not tokens or tokens[0] in {"OPENQASM", "include", "qreg"}:
#             continue  # Skip metadata lines
#
#         gate, *args = tokens
#         if "(" in gate:  # Parameterized gate
#             gate_name, param = gate.split("(")
#             param = param.rstrip(");")
#             qubit = int(args[0].replace("q[", "").replace("];", ""))
#
#             if gate_name == "virt_z":
#                 qc.append(VirtualZGate(float(param)), [qubit])  # Apply custom Virtual-Z gate
#             elif gate_name == "GPI":
#                 qc.append(GPIGate(float(param)), [qubit])  # Apply custom GPI gate
#             elif gate_name == "GPI2":
#                 qc.append(GPI2Gate(float(param)), [qubit])  # Apply custom GPI2 gate
#             elif hasattr(qc, gate_name):  # Check if the gate exists in Qiskit
#                 getattr(qc, gate_name)(float(param), qubit)
#
#         elif gate == "cx":  # CNOT gate
#             q1 = int(args[0].replace("q[", "").replace("],", ""))
#             q2 = int(args[1].replace("q[", "").replace("];", ""))
#             qc.cx(q1, q2)
#
#     return qc  # Return the constructed QuantumCircuit


def load_qasm(file_path):
    """
    Load an OpenQASM 2.0 file and construct a Qiskit QuantumCircuit.

    This function:
    - Reads a QASM file line by line.
    - Parses standard gates like `rx`, `ry`, `rz`, `cx`, etc.
    - Replaces `GPI`, `GPI2`, `virt_z`, `fullMS`, and `partialMS` with their custom gate definitions.

    Args:
        file_path (str): Path to the QASM file.

    Returns:
        QuantumCircuit: A Qiskit circuit containing the parsed gates.
    """

    # Read the QASM file
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Determine the number of qubits from qreg
    num_qubits = 0
    for line in lines:
        if "qreg q[" in line:
            num_qubits = int(line.split("[")[1].split("]")[0])

    # Initialize a QuantumCircuit
    qc = QuantumCircuit(num_qubits)

    # Parse QASM lines
    for line in lines:
        tokens = line.strip().split()
        if not tokens or tokens[0] in {"OPENQASM", "include", "qreg"}:
            continue  # Skip metadata lines

        gate, *args = tokens
        if "(" in gate:  # Parameterized gate
            gate_name, param = gate.split("(")
            #TWO qubit parametrized
            if gate_name == "fullMS":
                q1 = int(args[1].replace("q[", "").replace("],", "").replace("];", ""))
                q2 = int(args[2].replace("q[", "").replace("];", ""))
                phi1=float(param.split(",")[0])
                phi2=float(args[0].split(")")[0])
                qc.append(FullMSGate(phi1,phi2 ), [q1,q2])
            elif gate_name == "partialMS":
                q1 = int(args[2].replace("q[", "").replace("],", "").replace("];", ""))
                q2 = int(args[3].replace("q[", "").replace("];", ""))
                phi1=float(param.split(",")[0])
                phi2=float(args[0].split(",")[0])
                theta=float(args[1].split(")")[0])
                qc.append(PartialMSGate(phi1,phi2,theta), [q1,q2])
            else:
                param = param.rstrip(");")
                qubit_args = [int(q.replace("q[", "").replace("],", "").replace("];", "")) for q in args]

                if gate_name == "virt_z":
                    qc.append(VirtualZGate(float(param)), [qubit_args[0]])  # Apply custom Virtual-Z gate
                elif gate_name == "GPI":
                    qc.append(GPIGate(float(param)), [qubit_args[0]])  # Apply custom GPI gate
                elif gate_name == "GPI2":
                    qc.append(GPI2Gate(float(param)), [qubit_args[0]])  # Apply custom GPI2 gate
                elif hasattr(qc, gate_name):  # Check if the gate exists in Qiskit
                    getattr(qc, gate_name)(float(param), qubit_args[0])

        elif gate == "cx":  # CNOT gate
            q1 = int(args[0].replace("q[", "").replace("],", "").replace("];", ""))
            q2 = int(args[1].replace("q[", "").replace("];", ""))
            qc.cx(q1, q2)

    return qc  # Return the constructed QuantumCircuit

# # Load the QASM file and construct the Qiskit circuit
qc = load_qasm("circuit.qasm")
# Print the circuit as ASCII text (Qiskit's built-in drawer)
print(qc.draw(output='text'))
