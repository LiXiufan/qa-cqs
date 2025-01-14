########################################################################################################################
# Copyright (c) Xiufan Li. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Xiufan Li
# Supervisor: Patrick Rebentrost
# Institution: Centre for Quantum Technologies, National University of Singapore
# For feedback, please contact Xiufan at: shenlongtianwu8@gmail.com.
########################################################################################################################

# !/usr/bin/env python3

"""
    Test the instance generation functions. Here we make sure the problem is well-defined,
    because we are creating Hermitian strict diagonally dominant matrix with only real and positive diagonal entries.
    This matrix is positive definite and thus invertible, making us easier for analysis.
"""

from cqs.object import Instance, RandomInstance
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# First type of generation: given coefficients, given unitary, given ub.
n1 = 5
K1 = 3
instance1 = Instance(n1, K1)
coeffs1 = [1, 0.2, 0.5]
print("The first problem has coefficients:", coeffs1)
unitaries1 = [[['I', 'I', 'I', 'I', 'I']], [['X', 'Z', 'I', 'I', 'I']], [['Z', 'Y', 'I', 'I', 'I']]]
print("The first problem has unitaries:", unitaries1)
ub1 = [['I', 'I', 'I', 'I', 'I']]
print("The first problem has a unitary b:", ub1)
instance1.generate(given_coeffs=coeffs1, given_unitaries=unitaries1, given_ub=ub1)
matrix1 = instance1.get_matrix()
print("The first example returns with a matrix:")
print(matrix1)
print()


# The following ways of input is used for our scalability test.
# Second type of generation: random coefficients, random Pauli strings, given ub.
n2 = 5
K2 = 3
ub2 = [['I', 'I', 'I', 'I', 'I']]
instance2 = RandomInstance(n2, K2)
instance2.generate(given_ub=ub2) # which_type = 'Pauli_eigens'
coeffs2 = instance2.get_coeffs()
print("The second problem has coefficients:", coeffs2)
unitaries2 = instance2.get_unitaries()
print("The second problem has unitaries:", unitaries2)
print("The second problem has a unitary b:", ub2)
matrix2 = instance2.get_matrix()
print("The second example returns with a matrix:")
print(matrix2)
print()


# Third type of generation: random coefficients, random Pauli strings, random Pauli ub.
n3 = 3
K3 = 3
instance3 = RandomInstance(n3, K3)
instance3.generate() # which_type = 'Pauli_eigens'
coeffs3 = instance3.get_coeffs()
print("The third problem has coefficients:", coeffs3)
unitaries3 = instance3.get_unitaries()
print("The third problem has unitaries:", unitaries3)
ub3 = instance3.get_ub()
print("The third problem has a unitary b:", ub3)
matrix3 = instance3.get_matrix()
print("The third example returns with a matrix:")
print(matrix3)
print()


# Fourth type of generation: random coefficients, random Pauli gates, random Pauli ub gate.
n4 = 3
K4 = 3
instance4 = RandomInstance(n4, K4)
instance4.generate(which_type='Pauli_gates')
coeffs4 = instance4.get_coeffs()
print("The fourth problem has coefficients:", coeffs4)
unitaries4 = instance4.get_unitaries()
print("The fourth problem has unitaries:")
for u in unitaries4:
    print(u)
ub4 = instance4.get_ub()
print("The fourth problem has a unitary b:")
print(ub4)
matrix4 = instance4.get_matrix()
print("The fourth example returns with a matrix:")
print(matrix4)
print()


# Fifth type of generation: random coefficients, random unitaries, given ub unitary.
n5 = 3
K5 = 3
instance5 = RandomInstance(n5, K5)
qr = QuantumRegister(n5)
cr = ClassicalRegister(n5, 'c')
ub5 = QuantumCircuit(qr, cr)
ub5.h(0)
ub5.cx(0, 1)
ub5.cx(1,2)
instance5.generate(which_type='Haar', given_ub=ub5)
coeffs5 = instance5.get_coeffs()
print("The fifth problem has coefficients:", coeffs5)
unitaries5 = instance5.get_unitaries()
print("The fifth problem has unitaries:")
for u in unitaries5:
    print(u)
print("The fifth problem has a unitary b:")
print(ub5)
matrix5 = instance5.get_matrix()
print("The fifth example returns with a matrix:")
print(matrix5)
print()


# Sixth type of generation: random coefficients, random unitaries, random ub unitary.
n6 = 3
K6 = 3
instance6 = RandomInstance(n6, K6)
instance6.generate(which_type='Haar')
coeffs6 = instance6.get_coeffs()
print("The sixth problem has coefficients:", coeffs6)
unitaries6 = instance6.get_unitaries()
print("The sixth problem has unitaries:")
for u in unitaries6:
    print(u)
ub6 = instance6.get_ub()
print("The sixth problem has a unitary b:")
print(ub6)
matrix6 = instance6.get_matrix()
print("The sixth example returns with a matrix:")
print(matrix6)
print()


