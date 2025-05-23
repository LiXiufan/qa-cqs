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
    Optimization module for solving the optimal combination parameters.
    In this module, we implement the CVXOPT package as an external resource package.
    CVXOPT is a free software package for convex optimization based on the Python programming language.
    Reference: https://cvxopt.org
    MIT Course: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

    CVXOPT Notation of a quadratic optimization problem:
                min    1/2  x^T P x + q^T x
          subject to   Gx  <=  h
                       Ax  =  b
"""

from cvxopt import matrix
from cvxopt.solvers import qp
from numpy import linalg, diag, multiply, real, imag
from numpy import array, ndarray
from numpy import transpose, matmul
from sympy import Matrix
from typing import List, Tuple
import torch
import torch.optim as optim

__all__ = [
    "solve_combination_parameters"
]


def solve_combination_parameters(Q: ndarray, r: ndarray, which_opt=None, reg=0) -> Tuple[float, List]:
    r"""Optimization module for solving the optimal combination parameters.

    In this module, we implement the CVXOPT package as an external resource package.
    CVXOPT is a free software package for convex optimization based on the Python programming language.
    Reference: https://cvxopt.org
    MIT Course: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

    CVXOPT Notation of a quadratic optimization problem:
                min    1/2  x^T Q x + q^T r
          subject to   Gx  <=  h
                       Ax  =  b

    Args:
        Q (np.ndarray): the auxiliary matrix Q
        r (np.ndarray): the auxiliary vector r

    Returns:
        Tuple[float, List]: loss and the optimal combination parameters
    """
    if which_opt is None:
        which_opt = 'cvxopt'

    if which_opt == 'cvxopt':
        Q = 2 * matrix(Q)
        r = (-2) * matrix(r)
        # Solve the optimization problem using the kkt solver with regularization constant of 1e-12
        # Note: for more realistic experiments, due to the erroneous results,
        # it is suggested to change the regularization constant to get a better performance.
        # comb_params = qp(Q, r, kktsolver='ldl', options={'kktreg': 1e-16})['x']
        comb_params = qp(Q, r, kktsolver='ldl', options={'kktreg': 1e-2})['x']



    elif which_opt == "ADAM":

        Q = 2 * torch.Tensor(Q)

        r = (-2) * torch.Tensor(r)

        # Define the loss function L(x) = x^T Q x + r^T x

        def loss_function(x, Q, r):

            return torch.abs(0.5 * torch.matmul(x.T, torch.matmul(Q, x)) + torch.matmul(r.T, x) + 1) + reg * torch.matmul(x.T, x)

        # Initialize x, Q, and r

        dim = len(r)  # Dimension of x

        comb_params = torch.randn(dim, requires_grad=True)  # Trainable variable

        # Define optimizer

        optimizer = optim.Adam([comb_params], lr=0.02)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)  # Reduce LR every 1000 epochs

        max_epochs = 10 ** 6

        patience_epochs = 1500  # Check progress every 1500 epochs

        threshold = 0.95  # Loss must be at most 95% of the previous one

        # Training loop

        best_loss = float("inf")  # Keep track of the best loss

        last_improvement_epoch = 0  # Track last improvement

        for epoch in range(max_epochs):

            optimizer.zero_grad()

            loss = loss_function(comb_params, Q, r)  # Compute loss

            loss.backward()  # Compute gradients

            optimizer.step()  # Update x

            scheduler.step()  # Adjust learning rate

            # Check stopping criterion

            if loss.item() < best_loss * threshold:
                best_loss = loss.item()

                last_improvement_epoch = epoch  # Reset improvement tracker

            if epoch - last_improvement_epoch >= patience_epochs:
                # print(f"Stopping early at epoch {epoch + 1}, loss stagnated at {loss.item():.6f}")

                break
        comb_params = comb_params.tolist()
        # Train x
    elif which_opt == 'inv':
        Q = Matrix(Q)
        P, D = Q.diagonalize()
        D = array(D, dtype='complex128')
        D_diag = diag(D)
        D_diag_inv = []
        for d in D_diag:
            if linalg.norm(d) <= 1e-12:
                d_inv = 0
            else:
                if linalg.norm(real(d)) <= 1e-12:
                    d = 0 + imag(d) * 1j
                if linalg.norm(imag(d)) <= 1e-12:
                    d = real(d)
                d_inv = 1 / d
            D_diag_inv.append(d_inv)
        D_diag_inv = array(D_diag_inv, dtype='complex128')
        comb_params = multiply(D_diag_inv, r.reshape(-1))
    else:
        raise ValueError

    half_var = int(len(comb_params) / 2)
    alphas = [0 for _ in range(half_var)]

    for i in range(half_var):
        alpha = comb_params[i] + comb_params[half_var + i] * 1j
        alphas[i] = alpha

    # Calculate l2-norm loss function
    params_array = array(comb_params).reshape(-1, 1)
    Q_array = array(Q)
    r_array = array(r).reshape(-1, 1)
    loss = abs((matmul(matmul(transpose(params_array), 0.5*Q_array), params_array)
                + matmul(transpose(r_array), params_array) + 1).item())
    # Calculate Hamiltonian loss fucntion
    # loss = abs((transpose(params_array) @ Q_array @ params_array - (transpose(params_array) @ r_array) * transpose(r_array) @ params_array).item())
    return loss, alphas
