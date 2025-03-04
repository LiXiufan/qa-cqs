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
    Parameters initialization
"""

def set_params(backend=None,
               shots=None,
               device=None,
               optimization_level=None,
               noise_level_two_qubit=None,
               noise_level_one_qubit=None,
               readout_error=None):
    if backend is None:
        backend = 'eigens'
    if shots is None:
        shots = 1024
    if device is None:
        device = "Aria"
    if optimization_level is None:
        optimization_level = 1
    if noise_level_two_qubit is None:
        noise_level_two_qubit = 0
    if noise_level_one_qubit is None:
        noise_level_one_qubit = 0
    if readout_error is None:
        readout_error = 0
    return backend, shots, device, optimization_level, noise_level_two_qubit, noise_level_one_qubit, readout_error