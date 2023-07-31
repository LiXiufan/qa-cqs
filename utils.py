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
    Basic functions and usage.
"""

from typing import List, Tuple, Dict
from numpy import array, ndarray, random, pi, log
from numpy import transpose, conj
from numpy import matmul as mat
from Error import ArgumentError
from networkx import Graph, spring_layout, draw_networkx


import matplotlib.pyplot as plt

__all__ = [
    "PauliStrings"
]

ModuleErrorCode = 0
FileErrorCode = 0


def PauliStrings(which_Pauli):
    if which_Pauli == 'I':
        return array([[1, 0], [0, 1]], dtype='complex128')

    elif which_Pauli == 'X':
        return array([[0, 1], [1, 0]], dtype='complex128')

    elif which_Pauli == 'Y':
        return array([[0, -1j], [1j, 0]], dtype='complex128')

    elif which_Pauli == 'Z':
        return array([[1, 0], [0, -1]], dtype='complex128')


def draw_ansatz_tree(A_terms_number, current_tree_depth, which_process, which_index=None):
    plt.figure()
    plt.ion()
    plt.cla()
    plt.title("Ansatz Tree", fontsize=15)
    plt.xlabel("Tree Nodes (RED)  Optimal Child Node (GREEN)  Candidates (BLUE)", fontsize=12)
    plt.grid()
    # mngr = plt.get_current_fig_manager()
    # mngr.window.setGeometry(500, 100, 800, 600)
    colors = ['tab:red', 'tab:blue', 'tab:green']

    if current_tree_depth == 1:
        tree_nodes = [(0, 0)]
    else:
        tree_nodes = [(0, - i) for i in range(current_tree_depth - 1)]

    child_nodes = [(i - A_terms_number, - current_tree_depth) for i in range(A_terms_number)]
    e_tree = [(i, j) for i in tree_nodes for j in tree_nodes if i[1] > j[1]]
    e_expan = [(i, j) for i in child_nodes for j in child_nodes if i[0] < j [0]]

    if which_process == 'Expansion':
        V = tree_nodes + child_nodes
        E = e_tree + e_expan
        G = Graph()
        G.add_nodes_from(V)
        G.add_edges_from(E)
        position = {v: [v[0], - v[1]] for v in list(G.nodes)}
        nodes = [tree_nodes, child_nodes]
        for i in range(2):
            for v in nodes[i]:
                options = {
                    "nodelist": [v],
                    "node_color": colors[i],
                    "with_labels": False,
                    "width": 3,
                 }
            draw_networkx(G, position, **options)
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("on")
            ax.set_axisbelow(True)

    elif which_process == 'Optimal':
        V = tree_nodes + child_nodes
        E = e_tree + e_expan
        G = Graph()
        G.add_nodes_from(V)
        G.add_edges_from(E)
        position = {v: [v[0], - v[1]] for v in list(G.nodes)}
        which_node = child_nodes[which_index]
        nodes = [tree_nodes, child_nodes.remove(which_node), [which_node]]
        for i in range(3):
            for v in nodes[i]:
                options = {
                    "nodelist": [v],
                    "node_color": colors[i],
                    "with_labels": False,
                    "width": 3,
                }
            draw_networkx(G, position, **options)
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("on")
            ax.set_axisbelow(True)

    elif which_process == 'Pending':
        V = tree_nodes
        E = e_tree
        G = Graph()
        G.add_nodes_from(V)
        G.add_edges_from(E)
        position = {v: [v[0], - v[1]] for v in list(G.nodes)}
        options = {
            "nodelist": V,
            "node_color": colors[0],
            "with_labels": False,
            "width": 3,
        }
        draw_networkx(G, position, **options)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("on")
        ax.set_axisbelow(True)

    plt.pause(0.1)

    # for j in range(4):
    #     for vertex in vertex_sets[j]:
    #         options = {
    #             "nodelist": [vertex],
    #             "node_color": colors[j],
    #             "node_shape": '8' if vertex in ancilla_qubits else 'o',
    #             "with_labels": False,
    #             "width": 3,
    #         }
    #         draw_networkx(self.__graph, self.__pos, **options)
    #         ax = plt.gca()
    #         ax.margins(0.20)
    #         plt.axis("on")
    #         ax.set_axisbelow(True)
    # plt.pause(self.__pause_time)