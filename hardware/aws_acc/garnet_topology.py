# Plot the figure of ionq with different qubits
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()

n = 20
V = list(range(n))
color = ['red' for _ in range(3)] + ['green' for _ in range(n - 3)]

G.add_nodes_from(V)
G.add_edges_from([
    (0, 1), (0, 3), (1, 4), (2, 3),
    (2, 7), (3, 4), (3, 8), (4, 5), (4, 9),
    (5, 6), (5, 10), (6, 11), (7, 8), (7, 12),
    (8, 9), (8, 13), (9, 10), (9, 14), (10, 11),
    (10, 15), (11, 16), (12, 13), (13, 14), (13, 17),
    (14, 15), (14, 18), (15, 19), (15, 16),
    (17, 18), (18, 19)
])
nx.draw(G, with_labels=True, node_color=color, pos=nx.spring_layout(G), width=0.6)
plt.show()
