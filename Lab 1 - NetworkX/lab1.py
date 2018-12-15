# -*- coding: utf-8 -*-
"""
ST 2: Mathematical Modeling of Propagation Phenomena
TD 1: Graph Exploration 
10 December 2018
"""
from helper import *
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import networkx as nx

## Exercise 1.1
# Read the edgelist of the NetScience network
G = nx.read_edgelist("./NetScience.edgelist", comments='#', delimiter='\t')
## Exercise 1.2
def compute_network_characteristics(graph):
    prop = {}
    prop['N'] = len(G.nodes) #number of vertices
    prop['M'] =  len(G.edges)# number of edges

    prop['min_degree'] = min([degree for (node,degree) in G.degree])# minimum degree
    prop['max_degree'] =  max([degree for (node,degree) in G.degree]) # maximum degree
    prop['mean_degree'] = np.mean([degree for (node,degree) in G.degree]) # mean of node degrees
    prop['median_degree'] = np.median([degree for (node,degree) in G.degree]) # median of node degrees
    prop['density'] = 2*prop['M']/(prop['M']*(prop['M']-1))# density of the graph
    return prop

###############################################################################
###############################################################################

prop = compute_network_characteristics(graph=G)

print("Number of nodes: {}".format(prop['N']))
print("Number of edges: {}".format(prop['M']))
print("Min. degree: {}".format(prop['min_degree']))
print("Max. degree: {}".format(prop['max_degree']))
print("Mean degree: {}".format(prop['mean_degree']))
print("Median degree: {}".format(prop['median_degree']))
print("Density: {}".format(prop['density']))

## Exercise 2.1-2.3
def get_gcc(G):
    # Is the given graph connected?
    connected =  nx.is_connected(G)# check if the graph is connected or not
    if connected:
        print("The graph is connected")
        return graph
    else:
        print("The graph is not connected")

    # Find the number of connected components
    num_of_cc = nx.number_connected_components(G)
    print("Number of connected components: {}".format(num_of_cc))

    # Get the largest connected component (GCC) subgraph
    gcc = sorted(nx.connected_component_subgraphs(G, copy = True), key = len, reverse = True)[0]  #max([len(sub_graph) for sub_graph in nx.connected_components(G)])
    node_fraction = len(gcc.nodes)/len(G.nodes)
    edge_fraction =  len(gcc.edges)/len(G.edges)

    print("Fraction of nodes in GCC: {:.3f}".format(node_fraction))
    print("Fraction of edges in GCC: {:.3f}".format(edge_fraction))

    return gcc

# Get the GGC of the network
gcc = get_gcc(G)

## Exercise 2.4
# Visualize the GCC of the network
visualize(graph=gcc, values=gcc.degree(), node_size=10)


## Exercise 3.1
def count_triangles_of(graph, node):
    count = 0
    node_neighbors = nx.neighbors(graph,node)
    for neighbor in node_neighbors:
        for neighbor2 in nx.neighbors(graph,neighbor):
            if neighbor2 == node:
                count += 1
    return count

## Exercise 3.2
def count_all_triangles(graph):
    count = 0
    for vertice in graph.nodes:
        count += count_triangles_of(graph,vertice)
    count = count/3 #we gotta divide by three because for each triangle, we have 3 nodes in it
    return count

# Check if our implementation returns the correct value by comparing it with
# the built-in function method
#print(count_all_triangles(graph=G))
#print(np.sum(list(nx.triangles(G).values()))/3)
num_of_triangles = count_all_triangles(graph=G)
assert num_of_triangles == np.sum(list(nx.triangles(G).values()))/3, "Incorrect result!"
print("Number of triangles in the graph: {}".format(num_of_triangles))

