import argparse
import os
import pdb

import numpy as np
import networkx as nx
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import modularity

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


def get_karate_network_data():
    graph = nx.karate_club_graph()

    node_comm = np.array([graph.nodes[v]['club'] for v in graph])

    return nx.Graph(graph), node_comm


def get_graph(dataset):
    graph = nx.read_gml(f'real-classic/{dataset}.gml', label='id')
    node_comm = np.array([graph.nodes[v]['value'] for v in graph])

    largest_cc = max(nx.connected_components(graph), key=len)

    graph = graph.subgraph(largest_cc)

    node_comm = node_comm[np.array(list(largest_cc)) - 1]

    return nx.Graph(graph), node_comm


def get_node_label_network_data(dataset):

    adj_list = np.load(f'./ind.{dataset}.graph')
    labels = np.load(f'./ind.{dataset}.ally')

    graph = nx.from_dict_of_lists(adj_list).subgraph(range(len(labels)))

    largest_cc = max(nx.connected_components(graph), key=len)

    graph = graph.subgraph(largest_cc)

    comm = np.argmax(labels, axis=1)[list(largest_cc)]

    return graph, comm


def get_lfr_network_data(n, tau1, tau2, mu):
    graph = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)

    comm = list({frozenset(graph.nodes[v]['community']) for v in graph})

    node_comm = np.empty(n)

    for i, c in enumerate(comm):
        for node in c:
            node_comm[node] = i

    return graph, node_comm


def get_greedy_modularity_communities(graph):

    comm = greedy_modularity_communities(graph)

    mod = modularity(graph, comm)

    node_gr_comm = np.zeros(max(graph.nodes) + 1)

    for i, c in enumerate(comm):
        for node in c:
            node_gr_comm[node] = i + 1

    return node_gr_comm[list(graph.nodes)], mod


def get_girvan_newman_communities(graph):

    dendogram = girvan_newman(graph)

    comm = tuple(sorted(c) for c in next(dendogram))

    mod = modularity(graph, comm)

    node_gn_comm = np.zeros(max(graph.nodes) + 1)

    for i, c in enumerate(comm):
        for node in c:
            node_gn_comm[node] = i + 1

    return node_gn_comm[list(graph.nodes)], mod


def get_clique_communities(graph, k=4):

    comm = list(k_clique_communities(graph, k))

    try:
        mod = modularity(graph, comm)
    except:
        mod = 0

    node_cl_comm = np.zeros(max(graph.nodes) + 1)

    for i, c in enumerate(comm):
        for node in c:
            node_cl_comm[node] = i + 1

    return node_cl_comm[list(graph.nodes)], mod


def main(params):

    if params.type == 'lfr':
        graph, comm = get_lfr_network_data(250, 3, 1.5, 0.1)
    elif params.type == 'node':
        graph, comm = get_node_label_network_data(params.dataset)
    elif params.type == 'real':
        graph, comm = get_graph(params.dataset)
    elif params.type == 'karate':
        graph, comm = get_karate_network_data()

    print(f'Number of nodes in the graph = {len(graph)}')
    print('The components of the graph are as follows -- ')
    print([len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)])

    greedy_comm, greedy_mod = get_greedy_modularity_communities(graph)
    if params.dataset != 'pubmed':

    gn_comm, gn_mod = get_girvan_newman_communities(graph)
    if params.dataset != 'polblogs':
        clique_comm, clique_mod = get_clique_communities(graph)

    with open(f'{params.dataset}.txt', "w+") as f:
        f.write(f"Greedy -- NMI={normalized_mutual_info_score(comm, greedy_comm, average_method='arithmetic')}; ARI={adjusted_rand_score(comm, greedy_comm)}; Modularity={greedy_mod}\n")
        if params.dataset != 'pubmed':
            f.write(f"Girvan-Newman -- NMI={normalized_mutual_info_score(comm, gn_comm, average_method='arithmetic')}; ARI={adjusted_rand_score(comm, gn_comm)}; Modularity={gn_mod}\n")
        if params.dataset != 'polblogs':
            f.write(f"Clique -- NMI={normalized_mutual_info_score(comm, clique_comm, average_method='arithmetic')}; ARI={adjusted_rand_score(comm, clique_comm)}; Modularity={clique_mod}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network data metrics')

    parser.add_argument("--dataset", "-d", type=str, default="citeseer",
                        help="Dataset name")
    parser.add_argument("--type", "-t", type=str, default="real",
                        help="Dataset name")

    params = parser.parse_args()

    main(params)
