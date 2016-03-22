import matplotlib.pyplot as plt
import networkx as nx

G=nx.complete_graph(3)
nx.draw(G)

plt.savefig("/home/ggdhines/github/aggregation/docs/images/rectangle_graph.jpg")