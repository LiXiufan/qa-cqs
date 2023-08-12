from examples.CQS.cqs_main import EXE

qubit_number = 5
dim = 2 ** qubit_number
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 3
ITR = 1

# Total Budget of shots
shots_total_budget = 10 ** 6
# Expected error
error = 0.1

# Initialize the coefficient matrix
# print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

# Generate A with the following way
# This demo is taken from the VQSL article
coeffs = [1, 0.2, 0.2]
unitaries = [[['I', 'I', 'I', 'I', 'I']], [['X', 'Z', 'I', 'I', 'I']], [['X', 'I', 'I', 'I', 'I']]]
u_b = [['H', 'I', 'H', 'H', 'H']]
# A_norm = 5.878775382679628
# coeffs_normalized = [i / A_norm for i in coeffs]

file_name = 'cqs_exe_first_demo.txt'
# Number of Hadamard tests in total: 636
# So the total shot budget is: 636 * 11 =  6996

# coeffs =  [0.358, 0.011, -0.919, -0.987]
# unitaries = [[['I', 'Y', 'X', 'I', 'Y']], [['Z', 'X', 'Y', 'X', 'I']], [['X', 'X', 'Z', 'I', 'X']], [['Z', 'I', 'X', 'Z', 'Z']]]

EXE(qubit_number, number_of_terms, ITR, coeffs, unitaries, u_b, file_name, backend='matrix', expan_mtd='gradient')







