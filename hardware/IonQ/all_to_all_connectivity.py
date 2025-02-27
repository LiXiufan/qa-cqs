# Plot the figure of ionq with different qubits
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()

N = [11, 23, 32]
name = ['ionq Harmony', 'ionq Aria', 'ionq Forte']
for k in range(3):
    n = N[k]
    V = list(range(n))
    color = ['red' for _ in range(3)] + ['green' for _ in range(n - 3)]

    G.add_nodes_from(V)

    E = []
    for i in range(n):
        for j in range(i + 1, n):
            E.append((i, j))

    G.add_edges_from(E)
    nx.draw(G, with_labels=True, node_color=color, pos=nx.circular_layout(G), width=0.6)

    # plt.figtext(0.5, 0.01, 'green: pending qubits;                  red: activating qubits', ha="center", fontsize=24)
    # plt.title(name[k])
    plt.show()
