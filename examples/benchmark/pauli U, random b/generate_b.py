from qiskit.circuit.random import random_circuit
n = 2
d = 2
a = random_circuit(n, d, measure=False)

print(a)
