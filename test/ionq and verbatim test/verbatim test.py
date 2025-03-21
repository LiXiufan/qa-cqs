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

r"""
   This module tests the verbatim execution of quantum tasks.
"""
from transpiler.qasm2_reader import from_qasm2_to_braket
from braket.aws import AwsDevice

circuit_qasm = from_qasm2_to_braket('circuit.qasm')
print(circuit_qasm)

# Try with Aria-1
device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")

# Choose S3 bucket to store results
bucket = 'amazon-braket-prgroup-xiufan'# <<Update with your actual bucket name>> #eg: "amazon-braket-unique-aabbcdd"
prefix = "results"
s3_folder = (bucket, prefix)

task = device.run(circuit_qasm, shots=2, disable_qubit_rewiring=True)
result = task.result()
print("Measurement Results:", result.measurement_counts)



