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
    Benchmark problem sizes and algorithm scalability.
"""
from cqs.object import RandomInstance
from random import choice
from numpy import linalg
from cqs.calculation import calculate_Q_r_by_eigens
from cqs.optimization import solve_combination_parameters
from cqs.expansion import expand_ansatz_tree_by_eigens
nList = list(range(2, 10))
KList = list(range(2, 10))
ITR = 10

n = choice(nList)
K = choice(KList)
# n = 3
# K = 3
instance1 = RandomInstance(n, K)
instance1.generate()
coeffs = instance1.get_coeffs()
unitaries = instance1.get_unitaries()
ub = instance1.get_ub()
# problem statement
print("Coefficients are:", coeffs)
print("Unitaries are:", unitaries)
print("Vector b is:", ub)

ansatz_tree = [ub]
for itr in range(1, ITR + 1):
    print(ansatz_tree)
    Q, r = calculate_Q_r_by_eigens(instance1, ansatz_tree)
    # print("Auxiliary matrix Q:")
    # print(Q)
    # print("Auxiliary vector r:")
    # print(r)

    loss, alphas = solve_combination_parameters(Q, r)
    print("Iter:", itr)
    print("Loss:", loss)
    print("Alphas:", alphas)
    if itr < ITR:
        ansatz_tree = expand_ansatz_tree_by_eigens(instance1, alphas, ansatz_tree, mtd=None)












