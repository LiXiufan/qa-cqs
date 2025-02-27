# import utils
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider
# import gates
from qiskit_ionq import GPIGate, GPI2Gate, MSGate

# initialize a quantum circuit
circuit = QuantumCircuit(2, 2)
# add gates
circuit.append(MSGate(0, 0), [0, 1])
circuit.append(GPIGate(0), [0])
circuit.append(GPI2Gate(1), [1])
circuit.measure([0, 1], [0, 1])
print(circuit)



# Transpiling a circuit to native gates
qc_abstract = QuantumCircuit(3, 3, name="hello world, native gates")
qc_abstract.h(0)
qc_abstract.cx(0, 1)
qc_abstract.cx(1, 2)
qc_abstract.cx(1, 2)
qc_abstract.cx(1, 2)
qc_abstract.cx(0, 1)
qc_abstract.h(0)
qc_abstract.h(1)
qc_abstract.h(2)
qc_abstract.measure([0, 1, 2], [0, 1, 2])
print(qc_abstract)
provider = IonQProvider()
# provider = IonQProvider('pUhwyKCHRYAvWUChFqwTApQwow4mS2h7')
# provider = IonQProvider('0qWvYSerhtLhusDQfgqUd5zVSNyFbCRb')

backend_native = provider.get_backend("simulator", gateset="native")
qc_native1 = transpile(qc_abstract, backend=backend_native, optimization_level=1)
qc_native2 = transpile(qc_abstract, backend=backend_native, optimization_level=2)
qc_native3 = transpile(qc_abstract, backend=backend_native, optimization_level=3)

print(qc_native1)
print(qc_native2)
print(qc_native3)




