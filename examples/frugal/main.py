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
    Test the shot frugal method for the summation of different expectations obtained by Hadamard tests.
    H = \sum_{k=1}^{K} \beta_k U_k
    Goal: Estimate the <b|H|b> with a limited number of shots.

    Assume: all betas are in [-1, 1]
"""
from examples.frugal.estimate import main

NRANGE = list(range(5, 11))
SHOTS = 10200
SAMPLE = 200
FILE = 'shotfrugalData.txt'
main(NRANGE, SHOTS, SAMPLE, FILE)


