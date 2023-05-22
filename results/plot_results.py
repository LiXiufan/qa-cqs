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
    This is the calculator module to compute Q and r according to the Hadamard test shooting outcomes.
"""

import matplotlib.pyplot as plt

plt.title("CQS: Loss - Depth - Ideal")
depth = 50
Depth = list(range(depth))
Loss_list = [1, 0.816446429,
0.718691443,
0.683807513,
0.668437042,
0.586574712,
0.585949,
0.552832078,
0.533166504,
0.527385053,
0.511464006,
0.500955539,
0.48886759,
0.484526798,
0.478594238,
0.477962862,
0.466658745,
0.466070433,
0.463489626,
0.462241909,
0.461675122,
0.458779551,
0.456188246,
0.453792696,
0.452152916,
0.4022613,
0.400972353,
0.394824003,
0.390298044,
0.38973532,
0.365680905,
0.364570764,
0.360164207,
0.358883115,
0.35811932,
0.357522421,
0.353621409,
0.353062901,
0.352510061,
0.350658407,
0.350492017,
0.349584852,
0.349382402,
0.349141241,
0.34840114,
0.344915422,
0.34372053,
0.342544957,
0.340249655,
0.338520035

]

plt.plot(Depth, [0 for _ in Depth], 'b--',
         Depth, Loss_list, 'r-')

plt.xlabel("Depth")
plt.ylabel("Loss")
plt.show()
