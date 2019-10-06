import argparse
import os
import pdb

import numpy as np
import networkx as nx
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import k_clique_communities

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


def get_graph(path):
    graph = nx.read_gml(path, label='id')
    node_comm = np.array([graph.nodes[v]['value'] for v in graph])

    return nx.Graph(graph), node_comm


def get_lfr_network_data(n, tau1, tau2, mu):
    graph = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)

    comm = list({frozenset(graph.nodes[v]['community']) for v in graph})

    node_comm = np.empty(n)

    for i, c in enumerate(comm):
        for j, node in enumerate(c):
            node_comm[j] = i

    return graph, node_comm


def get_greedy_modularity_communities(graph):

    comm = greedy_modularity_communities(graph)

    node_gr_comm = np.empty(len(graph))

    for i, c in enumerate(comm):
        for node in c:
            node_gr_comm[node - 1] = i

    return node_gr_comm


def get_girvan_newman_communities(graph):

    dendogram = girvan_newman(graph)

    comm = tuple(sorted(c) for c in next(dendogram))

    node_gn_comm = np.empty(len(graph))

    for i, c in enumerate(comm):
        for node in c:
            node_gn_comm[node - 1] = i

    return node_gn_comm


def get_clique_communities(graph, k=3):

    comm = list(k_clique_communities(graph, k))

    node_cl_comm = np.empty(len(graph))

    for i, c in enumerate(comm):
        for node in c:
            node_cl_comm[node - 1] = i

    return node_cl_comm


def main(params):

    # graph, comm = get_lfr_network_data(250, 3, 1.5, 0.1)
    # graph, comm = get_graph('real-classic/football.gml')
    graph, comm = get_graph('real-classic/polblogs.gml')
    # graph, comm = get_graph('real-classic/polbooks.gml')
    # graph, comm = get_graph('real-classic/strike.gml')

    print(f'Number of nodes in the graph = {len(graph)}')
    print('The components of the graph are as follows -- ')
    print([len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)])

    greedy_comm = get_greedy_modularity_communities(graph)
    gn_comm = get_girvan_newman_communities(graph)
    # clique_comm = get_clique_communities(graph)

    with open('polblogs.txt', "w+") as f:
        f.write(f"Greedy -- NMI={normalized_mutual_info_score(comm, greedy_comm, average_method='arithmetic')}; ARI={adjusted_rand_score(comm, greedy_comm)}\n")
        f.write(f"Girvan-Newman -- NMI={normalized_mutual_info_score(comm, gn_comm, average_method='arithmetic')}; ARI={adjusted_rand_score(comm, gn_comm)}\n")
        # f.write(f"Clique -- NMI={normalized_mutual_info_score(comm, clique_comm, average_method='arithmetic')}; ARI={adjusted_rand_score(comm, clique_comm)}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network data metrics')

    parser.add_argument("--dataset", "-d", type=str, default="metabolic",
                        help="Dataset name")

    params = parser.parse_args()

    main(params)
