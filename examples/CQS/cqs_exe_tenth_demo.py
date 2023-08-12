from examples.CQS.cqs_main import EXE

qubit_number = 3
dim = 2 ** qubit_number
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 2
ITR = 3

# Total Budget of shots
shots_total_budget = 10 ** 6
# Expected error
error = 0.1

# Initialize the coefficient matrix
# print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

# Generate A with the following way
# This demo is taken from the VQSL article
coeffs = [1, 1]
unitaries = [[['Z', 'Z', 'Z']], [['X', 'X', 'X']]]
u_b = [['I', 'I', 'I']]

file_name = '../../results/Quantum Devices/Original Data/cqs_exe_first_demo.txt'
# Number of Hadamard tests in total: 636
# So the total shot budget is: 636 * 11 =  6996

# coeffs =  [0.358, 0.011, -0.919, -0.987]
# unitaries = [[['I', 'Y', 'X', 'I', 'Y']], [['Z', 'X', 'Y', 'X', 'I']], [['X', 'X', 'Z', 'I', 'X']], [['Z', 'I', 'X', 'Z', 'Z']]]

EXE(qubit_number, number_of_terms, ITR, coeffs, unitaries, u_b, file_name)







