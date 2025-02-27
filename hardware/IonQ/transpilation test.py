from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from hardware.ionq.transpiler import transpile_to_ionq_native_gates

# qc = QuantumCircuit(3)
# qc.cx(0, 2)
# qc.x(0)
# qc.cz(0, 1)
# qc.cx(0, 2)
# qc.x(0)
# print(qc)


num_qubits = 5
ghz = QuantumCircuit(num_qubits)
ghz.h(range(num_qubits))
ghz.cx(0, range(1, num_qubits))
print("Pre-Transpilation: \n")
print(ghz)

transpiled_ghz = transpile_to_ionq_native_gates(ghz)
print("Post-Transpilation: \n")
print(transpiled_ghz)
