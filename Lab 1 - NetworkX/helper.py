#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ST 2: Mathematical Modeling of Propagation Phenomena
TD 1: Graph Exploration 
10 December 2018
"""
import matplotlib.pyplot as plt
import networkx as nx


def visualize(graph, values, node_size=100):
    '''
    :param graph:
    :param node_list: dictionary keyed by node
    :return:
    '''
    nodes = graph.nodes()
    colors = [values[node] for node in nodes]
    
    pos = nx.spring_layout(graph)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, 
                                with_labels=False, node_size=node_size, cmap=plt.cm.jet)
    plt.colorbar(nc)
    plt.axis('off')
    
    plt.show()
