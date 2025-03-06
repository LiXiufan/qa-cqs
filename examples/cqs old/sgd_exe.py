from cqs.expansion import expand_ansatz_tree
from cqs.object import CoeffMatrix
from cqs.expansion import optimize_with_stochastic_descend
from cqs.calculation import verify_loss_function
from numpy import array, log2, linalg

import matplotlib.pyplot as plt

number_of_qubits = 6
dim = 2 ** number_of_qubits
# Set the number of terms of the coefficient matrix A on the left hand side of the equation.
# According to assumption 1, the matrix A has the form of linear combination of known unitaries.
# For the near-term consideration, the number of terms are in the order of magnitude of ploy(log(dimension)).
number_of_terms = 20
shots = 5000
total_tree_depth = 20
error = 0.1

# Initialize the coefficient matrix
A = CoeffMatrix(number_of_terms, dim, number_of_qubits)
# print('Qubits are tagged as:', ['Q' + str(i) for i in range(A.get_width())])

# Generate A with Pauli matrices
A.generate()

# Generate A with other forms (Haar matrices)
# A.generate('Haar')

# Get the coefficients of the terms and the unitaries
coeffs = A.get_coeff()
unitaries = A.get_unitary()
# print('Coefficients of the terms are:', coeffs)
# print('Decomposed unitaries are:', unitaries)
B = sum([abs(coeff) for coeff in coeffs])

# Values on the right hand side of the equation.
# b = array([1] + [0 for _ in range(dim - 1)])
# print('The vector on the right hand side of the equation is:', b)

# N = 100
N_Total = 25
loss_sample = 200
plt.title("Stochastic Gradient Descend: Loss - Steps")


# for k in range(loss_sample):
#     Loss = []
#     for N in range(1, N_Total):
#         loss = optimize_with_stochastic_descend(A, N)
#         Loss.append(loss)
#     plt.plot(list(range(1, N_Total)), Loss, 'r-')

Loss = []
for N in range(1, N_Total):
    loss_list = []
    for k in range(loss_sample):
        loss = optimize_with_stochastic_descend(A, N)
        loss_list.append(loss)
    loss_ave = sum(loss_list) / loss_sample
    Loss.append(loss_ave)
plt.plot(list(range(1, N_Total)), Loss, 'r-')

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.show()






# print(x_opt)