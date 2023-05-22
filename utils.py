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


def print_progress(current_progress, progress_name, track=True):
    r"""画出当前步骤的进度条。

    Args:
        current_progress (float / int): 当前的进度百分比
        progress_name (str): 当前步骤的名称
        track (bool): 是否绘图的布尔开关

    代码示例：

    ..  code-block:: python

        from paddle_quantum.mbqc.utils import print_progress
        print_progress(14/100, "Current Progress")

    ::

       Current Progress              |■■■■■■■                                           |   14.00%
    """
    assert 0 <= current_progress <= 1, "'current_progress' must be between 0 and 1"
    assert isinstance(track, bool), "'track' must be a bool."
    if track:
        print(
            "\r"
            f"{progress_name.ljust(30)}"
            f"|{'■' * int(50 * current_progress):{50}s}| "
            f"\033[94m {'{:6.2f}'.format(100 * current_progress)}% \033[0m ", flush=True, end=""
        )
        if current_progress == 1:
            print(" (Done)")

def plot_results(dict_lst, bar_label, title, xlabel, ylabel, xticklabels=None):
    r"""根据字典的键值对，以键为横坐标，对应的值为纵坐标，画出柱状图。

    Note:
        该函数主要调用来画出采样分布或时间比较的柱状图。

    Args:
        dict_lst (list): 待画图的字典列表
        bar_label (list): 每种柱状图对应的名称
        title (str): 整个图的标题
        xlabel (str): 横坐标的名称
        ylabel (str): 纵坐标的名称
        xticklabels (list, optional): 柱状图中每个横坐标的名称
    """
    assert isinstance(dict_lst, list), "please input a list with dictionaries."
    assert isinstance(bar_label, list), "please input a list with bar_labels."
    assert len(dict_lst) == len(bar_label), \
        "please check your input as the number of dictionaries and bar labels are not equal."
    bars_num = len(dict_lst)
    bar_width = 1 / (bars_num + 1)
    plt.ion()
    plt.figure()
    for i in range(bars_num):
        plot_dict = dict_lst[i]
        # Obtain the y label and xticks in order
        keys = list(plot_dict.keys())
        values = list(plot_dict.values())
        xlen = len(keys)
        xticks = [((i) / (bars_num + 1)) + j for j in range(xlen)]
        # Plot bars
        plt.bar(xticks, values, width=bar_width, align='edge', label=bar_label[i])
        plt.yticks()
    if xticklabels is None:
        plt.xticks(list(range(xlen)), keys, rotation=90)
    else:
        assert len(xticklabels) == xlen, "the 'xticklabels' should have the same length with 'x' length."
        plt.xticks(list(range(xlen)), xticklabels, rotation=90)
    plt.legend()
    plt.title(title, fontproperties='SimHei', fontsize='x-large')
    plt.xlabel(xlabel, fontproperties='SimHei')
    plt.ylabel(ylabel, fontproperties='SimHei')
    plt.ioff()
    plt.show()


def write_running_data(textfile, eg, width, mbqc_time, reference_time):
    r"""写入电路模拟运行的时间。

    由于在许多电路模型模拟案例中，需要比较我们的 ``MBQC`` 模拟思路与 ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型模拟思路的运行时间。
    因而单独定义了写入文件函数。

    Hint:
        该函数与 ``read_running_data`` 函数配套使用。

    Warning:
        在调用该函数之前，需要调用 ``open`` 打开 ``textfile``；在写入结束之后，需要调用 ``close`` 关闭 ``textfile``。

    Args:
        textfile (TextIOWrapper): 待写入的文件
        eg (str): 当前案例的名称
        width (float): 电路宽度（比特数）
        mbqc_time (float): ``MBQC`` 模拟电路运行时间
        reference_time (float):  ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型运行时间
    """
    textfile.write("The current example is: " + eg + "\n")
    textfile.write("The qubit number is: " + str(width) + "\n")
    textfile.write("MBQC running time is: " + str(mbqc_time) + " s\n")
    textfile.write("Circuit model running time is: " + str(reference_time) + " s\n\n")


def read_running_data(file_name):
    r"""读取电路模拟运行的时间。

    由于在许多电路模型模拟案例中，需要比较我们的 ``MBQC`` 模拟思路与 ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型模拟思路的运行时间。
    因而单独定义了读取文件函数读取运行时间，将其处理为一个列表，
    列表中的两个元素分别为 ``Qiskit`` 或 ``UAnsatz`` 电路模型模拟思路运行时间的字典和 ``MBQC`` 模拟思路运行时间的字典。

    Hint:
        该函数与 ``write_running_data`` 函数配套使用。

    Args:
        file_name (str): 待读取的文件名

    Returns:
        list: 运行时间列表
    """
    bit_num_lst = []
    mbqc_list = []
    reference_list = []
    remainder = {2: bit_num_lst, 3: mbqc_list, 4: reference_list}
    # Read data
    with open(file_name, 'r') as file:
        counter = 0
        for line in file:
            counter += 1
            if counter % 5 in remainder.keys():
                remainder[counter % 5].append(float(line.strip("\n").split(":")[1].split(" ")[1]))

    # Transform the lists to dictionaries
    mbqc_dict = {i: mbqc_list[i] for i in range(len(bit_num_lst))}
    refer_dict = {i: reference_list[i] for i in range(len(bit_num_lst))}
    dict_lst = [mbqc_dict, refer_dict]
    return dict_lst

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