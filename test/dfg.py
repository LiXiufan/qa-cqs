from qiskit.circuit.random import random_circuit
from transpiler.Noisy_simulations import transpiler, get_noisy_counts



# ===========================
# TEST: Transpile a Qiskit circuit
# ===========================

# Create a Qiskit quantum circuit
n = 3
d = 2
a = random_circuit(n, d, max_operands=2)



# Print the transpiled circuit
print("TEST Qiskit Circuit:")
print(a)

# Transpile the Qiskit circuit to use MS gates
transpiled_qc = transpiler(a)
# GET NOISEless probabilities
print(get_noisy_counts(transpiled_qc, 0))
# Print the transpiled circuit
print("Transpiled Qiskit Circuit:")
print(transpiled_qc)
#GET NOISY probabilities
print(get_noisy_counts(transpiled_qc, 0.02))










