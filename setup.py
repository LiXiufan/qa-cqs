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

"""
  The setup script to install for python
"""

from __future__ import absolute_import

from pathlib import Path

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

SDK_VERSION = '1.0'
DESC = Path('./README.md').read_text(encoding='utf-8')

setup(
    name='CQS',
    version=SDK_VERSION,
    install_requires=[
        # PySDK
        'numpy>=1.17.3',
        'requests>=2.28.0',
        'bidict>=0.22.0',
        'websocket-client>=1.3.2',

        # Example
        'scipy>=1.8.0',
        'matplotlib>=3.3.0',
        'networkx',
        'sympy>=1.10.1',

        # Hardware
        'qiskit>=0.41.0',
        'qiskit-terra>=0.17.4',
        'qiskit-ionq',
        'qiskit-ibm-runtime',
        'qiskit-ibm-provider',
        'qibo',
        'qibojit',
        'cvxopt'
    ],
    python_requires='>=3.8, <3.11',
    packages=find_packages(),
    license='Apache License 2.0',
    author='Xiufan Li',
    author_email='shenlongtianwu8@gmail.com',
    description='CQS is a Python-based quantum software development kit (SDK). '
                'It provides a full-stack programming experience for testing the performance of '
                'near-term quantum algorithms for linear systems of equations via high-performance simulators, '
                "various hardware frameworks and a focus on IonQ's trapped-ion quantum computer.",
    long_description=DESC,
    long_description_content_type='text/markdown'
)
