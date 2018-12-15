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


###############################################################################
#################### Please complete the required fields ######################
###############################################################################
#%% 
## Exercise 1.1
# Read the edgelist of the NetScience network
G = nx.read_edgelist("./NetScience.edgelist", comments='#', delimiter='\t')
         
## Exercise 1.2
def compute_network_characteristics(graph):
    prop = {}
    prop['N'] = graph.number_of_nodes() # number of nodes
    prop['M'] = graph.number_of_edges() # number of edges
    degrees = graph.degree().values()
    #degrees = [degree for node, degree in graph.degree()]
    prop['min_degree'] = np.min(degrees) # minimum degree
    prop['max_degree'] = np.max(degrees) # maximum degree
    prop['mean_degree'] = np.mean(degrees) # mean of node degrees
    prop['median_degree'] = np.median(degrees) # median of node degrees
    prop['density'] = nx.density(graph) # density of the graph
    #density
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





#%% 
###############################################################################
#################### Please complete the required fields ######################
###############################################################################

## Exercise 2.1-2.3
def get_gcc(graph):
    # Is the given graph connected?
    connected = nx.is_connected(graph) # check if the graph is connected or not
    if connected:
        print("The graph is connected")
        return graph
    
    print("The graph is not connected")
    
    # Find the number of connected components
    num_of_cc = nx.number_connected_components(graph)
    print("Number of connected components: {}".format(num_of_cc))
    
    # Get the greatest connected component subgraph
    gcc = max(nx.connected_component_subgraphs(graph), key=len)
    node_fraction = gcc.number_of_nodes() / float(graph.number_of_nodes())
    edge_fraction = gcc.number_of_edges() / float(graph.number_of_edges())
    
    print("Fraction of nodes in GCC: {:.3f}".format(node_fraction))
    print("Fraction of edges in GCC: {:.3f}".format(edge_fraction))

    return gcc

###############################################################################
###############################################################################    
# Get the GGC of the network
gcc = get_gcc(graph=G)

## Exercise 2.4 
# Visualize the GCC of the network
#visualize(graph=gcc, values=gcc.degree(), node_size=10)
  

#%%
###############################################################################
#################### Please complete the required fields ######################
###############################################################################

## Exercise 3.1
def count_triangles_of(graph, node):
    count = 0
    for nb in nx.neighbors(graph, node):
        for nb_nb in nx.neighbors(graph, nb):
            if node in nx.neighbors(graph, nb_nb):
                count += 1            
    count = count / 2
    
    # Alternative solution
    count = 0
    for nb in nx.neighbors(graph, node):
        common_nb_count = len(list(nx.common_neighbors(graph, node, nb)))
        if common_nb_count > 0:
            count += common_nb_count
    count = count / 2
    
    return count

## Exercise 3.2
def count_all_triangles(graph):
    count = 0
      
    for node in graph.nodes():
        count += count_triangles_of(graph, node)
    count = count / 3
    
    ## Alternative solution
    return count

###############################################################################
###############################################################################

# Check if our implementation returns the correct value by comparing it with
# the built-in function method -- for both the whole graph and the gcc
num_of_triangles = count_all_triangles(graph=gcc)
assert num_of_triangles == np.sum(nx.triangles(gcc).values())/3, "Incorrect result!"
print("Number of triangles in the graph: {}".format(num_of_triangles))


#%%
###############################################################################
#################### Please complete the required fields ######################
###############################################################################

## Exercise 3.3
# Spectral computation of the number of triangles by using eigenvalues 
def spectral_num_of_triangles_counting(graph):
    
    A = nx.adjacency_matrix(graph).todense() # adjacency matrix of graph 
    eigval, eigvec = eigh(A) # compute eigenvalues
    count = np.sum(np.power(eigval, 3)) / 6 
    
    return count
    
###############################################################################
###############################################################################
# For simpliticy, compute the number of triangles in the GCC
spectral_count = spectral_num_of_triangles_counting(graph=gcc)
print("The number of triangles found by spectral method: {}".format(spectral_count))
print("The total number of triangles: {}".
      format(np.sum(nx.triangles(gcc).values())/3))


#%%
###############################################################################
#################### Please complete the required fields ######################
###############################################################################

## Exercise 4.1
# Complete the following method computing the clustering coefficient of a node
def clustering_coefficient_of(graph, node):
    cc = 0.0
    
    d = nx.degree(graph, node) # get the degree of the node
    if d > 1: # if the degree is greater or equal to 2
        # Compute clustering coefficient
        cc = 2.0*count_triangles_of(graph, node) / (d * (d-1))
    
    return cc

## Exercise 4.2
def average_clustering_coefficient(graph):
    avcc = 0.0
    
    for node in graph.nodes():
        avcc += clustering_coefficient_of(graph, node)
    avcc = avcc / graph.number_of_nodes()        
        
    return avcc

###############################################################################
###############################################################################
    
avg = average_clustering_coefficient(graph=G)
assert np.abs(avg - nx.average_clustering(G))<0.001, "Incorrect result!"
print("Average clustering coefficient: {:.3f}".format(avg))


#%%
###############################################################################
#################### Please complete the required fields ######################
###############################################################################

## Exercise 5.1
def plot_histograms(graph):
    # degree sequence
    degree_sequence = graph.degree().values()
    #degree_sequence = [degree for node, degree in graph.degree()] 
    degree_count_values = sorted(set(degree_sequence))
    degree_hist = [degree_sequence.count(val) for val in degree_count_values]
    
    # the sequence of number of triangles that each node participates
    triangle_participation = nx.triangles(graph).values() 
    tr_count_values = sorted(set(triangle_participation))
    tri_hist = [triangle_participation.count(val) for val in tr_count_values]

    plt.subplot(1,2,1)
    plt.bar(degree_count_values, degree_hist)
    plt.xlabel("Degrees")
    plt.ylabel("Count")
    plt.subplot(1,2,2)
    plt.bar(tr_count_values, tri_hist)
    plt.xlabel("Number of triangles")
    plt.ylabel("Count")
    
    plt.show()

plot_histograms(graph=G)

## Exercise 5.2
# Generate an Erdos-Renyi graph
#er_graph = nx.erdos_renyi_graph(n=300, p=0.4)
#plot_histograms(graph=er_graph)
###############################################################################  
###############################################################################

#%%
###############################################################################
#################### Please complete the required fields ######################
###############################################################################

## Exercise 6
def compute_diameter(graph):
    # It is assumed that given graph is connected
    diameter = 0
    
    max_lens = {}
    for node in graph.nodes():
        length=nx.single_source_shortest_path_length(G, node)
        max_lens[node] = max(length.values())

    diameter = max(max_lens.values())
    
    return diameter

###############################################################################
###############################################################################
diameter = compute_diameter(graph=gcc)
assert diameter == nx.diameter(gcc), "Incorrect result!"

print("Diameter of the gcc: {}".format(diameter)) 


