from qiskit import QuantumCircuit  # Import the QuantumCircuit class from Qiskit
from transpiler.qiskit_custom_gate_definitions import VirtualZGate,GPIGate,GPI2Gate,FullMSGate,PartialMSGate
import numpy as np
from braket.circuits import Circuit

#all parameters are supposed to be 2pi periodic
def normalize_param(param):
    new_param=param
    while new_param>2*np.pi:
        new_param-=2*np.pi
    while new_param<0:
        new_param+=2*np.pi
    return new_param

def normalize_theta(param):
    new_param=param
    while new_param>np.pi/2:
        new_param-=np.pi/2
    while new_param<0:
        new_param+=np.pi/2
    return new_param

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
                phi1=normalize_param(float(param.split(",")[0]))
                phi2=normalize_param(float(args[0].split(")")[0]))
                qc.append(FullMSGate(phi1,phi2 ), [q1,q2])
            elif gate_name == "partialMS":
                q1 = int(args[2].replace("q[", "").replace("],", "").replace("];", ""))
                q2 = int(args[3].replace("q[", "").replace("];", ""))
                phi1=normalize_param(float(param.split(",")[0]))
                phi2=normalize_param(float(args[0].split(",")[0]))
                theta=normalize_theta(float(args[1].split(")")[0]))
                qc.append(PartialMSGate(phi1,phi2,theta), [q1,q2])
            else:
                param = normalize_param(float(param.rstrip(");")))

                qubit_args = [int(q.replace("q[", "").replace("],", "").replace("];", "")) for q in args]

                if gate_name == "virt_z":
                    qc.append(VirtualZGate(param), [qubit_args[0]])  # Apply custom Virtual-Z gate
                elif gate_name == "GPI":
                    qc.append(GPIGate(param), [qubit_args[0]])  # Apply custom GPI gate
                elif gate_name == "GPI2":
                    qc.append(GPI2Gate(param), [qubit_args[0]])  # Apply custom GPI2 gate
                elif hasattr(qc, gate_name):  # Check if the gate exists in Qiskit
                    getattr(qc, gate_name)(param, qubit_args[0])

        elif gate == "cx":  # CNOT gate
            q1 = int(args[0].replace("q[", "").replace("],", "").replace("];", ""))
            q2 = int(args[1].replace("q[", "").replace("];", ""))
            qc.cx(q1, q2)

    return qc  # Return the constructed QuantumCircuit


def from_qasm2_to_braket(file_name: str):
    """
    Convert an OpenQASM 2.0 file to an Amazon Braket Circuit.

    Args:
        file_name (str): Path to the OpenQASM file.

    Returns:
        Circuit: A Braket Circuit containing the parsed gates.

    Supports only: GPI, GPI2, MS
    """
    with open(file_name, "r") as f:
        lines = f.readlines()

    num_qubits = 0
    for line in lines:
        if "qreg q[" in line:
            num_qubits = int(line.split("[")[1].split("]")[0])
            break

    braket_circuit = Circuit()

    for line in lines:
        tokens = line.strip().split()
        if not tokens or tokens[0] in {"OPENQASM", "include", "qreg"}:
            continue  # Skip metadata lines

        gate, *args = tokens

        if "(" in gate:  # Parameterized gate
            gate_name, param = gate.split("(")
            if gate_name == "partialMS":
                q1 = int(args[2].replace("q[", "").replace("],", "").replace("];", ""))
                q2 = int(args[3].replace("q[", "").replace("];", ""))
                phi1 = normalize_param(float(param.split(",")[0]))
                phi2 = normalize_param(float(args[0].split(",")[0]))
                theta = normalize_theta(float(args[1].split(")")[0]))
                braket_circuit.ms(q1, q2, theta, phi1, phi2)
            else:
                qubit = [int(q.replace("q[", "").replace("],", "").replace("];", "")) for q in args][0]
                param = normalize_param(float(param.rstrip(");")))

                if gate_name == "GPI":
                    braket_circuit.gpi(qubit, param)

                elif gate_name == "GPI2":

                    braket_circuit.gpi2(qubit, param)


    return Circuit().add_verbatim_box(braket_circuit)


if __name__ == "__main__":
    print(from_qasm2_to_braket("circuit.qasm"))

