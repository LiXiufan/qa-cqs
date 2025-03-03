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
from cqs.local.calculation import calculate_Q_r
from cqs.optimization import solve_combination_parameters
from cqs.local.expansion import expand_ansatz_tree_by_eigens

convergence_loss = 0.01
stopping_iteration = 5
slow_iteration = 30
too_slow_iteration = 50
divergence_loss = 0.9
slow_loss = 0.5

NRANGE = list(range(11, 14))
KRANGE = list(range(3, 15))
SAMPLES = 5
FILE = 'BenchmarkPauliData.txt'
file1 = open(FILE, "a")

for n in NRANGE:
    for K in KRANGE:
        print("n, K:", n, K)
        for _ in range(1, SAMPLES + 1):
            instance1 = RandomInstance(n, K)
            instance1.generate()
            coeffs = instance1.get_coeffs()
            unitaries = instance1.get_unitaries()
            ub = instance1.get_ub()
            # TODO: change the initial point of ansatz tree
            ansatz_tree = [ub]
            ITR = []
            LOSS = []
            itr = 0
            loss = 1
            while loss >= convergence_loss:
                itr += 1
                Q, r = calculate_Q_r(instance1, ansatz_tree, backend='eigens')
                loss, alphas = solve_combination_parameters(Q, r)
                ITR.append(itr)
                LOSS.append(loss)
                if itr >= stopping_iteration and loss >= divergence_loss:
                    break
                if itr >= slow_iteration and loss >= slow_loss:
                    loss = 0
                    break
                if itr >= too_slow_iteration:
                    loss = 0
                    break
                if loss >= convergence_loss:
                    ansatz_tree = expand_ansatz_tree_by_eigens(instance1, alphas, ansatz_tree, mtd=None)


            # if the program runs successfully, record everything including problem statement, ITR, and LOSS
            if loss < divergence_loss:
                file1.writelines(["n:", str(n), '\n'])
                file1.writelines(["K:", str(K), '\n'])
                file1.writelines(["Coefficients:", str(coeffs), '\n'])
                file1.writelines(["Unitaries:", str(unitaries), '\n'])
                file1.writelines(["Ub:", str(ub), '\n'])
                file1.writelines(["ITR:", str(ITR), '\n'])
                file1.writelines(["LOSS:", str(LOSS), '\n\n'])












