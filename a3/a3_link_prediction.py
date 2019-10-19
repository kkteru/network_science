import argparse
import os
import pdb

import numpy as np
import networkx as nx
from networkx.algorithms.community import LFR_benchmark_graph

from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from networkx.algorithms import node_classification


def sample_neg(graph, num_neg_links):
    n = list(graph.nodes)
    neg_links = []
    while len(neg_links) < num_neg_links:
        h, t = np.random.choice(n), np.random.choice(n)
        if not graph.has_edge(h, t):
            neg_links.append((h, t))

    return neg_links


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def get_karate_network_data():
    graph = nx.karate_club_graph()

    return nx.Graph(graph)


def get_graph(dataset):
    graph = nx.read_gml(f'real-classic/{dataset}.gml', label='id')
    largest_cc = max(nx.connected_components(graph), key=len)

    graph = nx.Graph(graph.subgraph(largest_cc))
    all_edges = list(graph.edges)

    idx = np.arange(len(all_edges))
    np.random.shuffle(idx)
    test_idx = idx[:int(0.2 * len(graph.edges))]

    pos_test_links = [all_edges[i] for i in test_idx]

    n_test = len(test_idx)

    neg_test_links = sample_neg(graph, num_neg_links=n_test)

    graph.remove_edges_from(pos_test_links)

    return graph, pos_test_links, neg_test_links


def get_node_label_network_data(dataset):

    adj_list = np.load(f'./ind.{dataset}.graph')

    graph = nx.from_dict_of_lists(adj_list)

    largest_cc = max(nx.connected_components(graph), key=len)

    graph = nx.Graph(graph.subgraph(largest_cc))

    all_edges = list(graph.edges)

    idx = np.arange(len(all_edges))
    np.random.shuffle(idx)
    test_idx = idx[:int(0.2 * len(graph.edges))]

    pos_test_links = [all_edges[i] for i in test_idx]

    n_test = len(test_idx)

    neg_test_links = sample_neg(graph, num_neg_links=n_test)

    graph.remove_edges_from(pos_test_links)

    return graph, pos_test_links, neg_test_links


def get_lfr_network_data(n, tau1, tau2, mu):
    graph = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=30, seed=10)

    all_edges = list(graph.edges)

    idx = np.arange(len(all_edges))
    np.random.shuffle(idx)
    test_idx = idx[:int(0.2 * len(graph.edges))]

    pos_test_links = [all_edges[i] for i in test_idx]

    n_test = len(test_idx)

    neg_test_links = sample_neg(graph, num_neg_links=n_test)

    graph.remove_edges_from(pos_test_links)

    return graph, pos_test_links, neg_test_links


def get_link_pred_auc(graph, pos_test, neg_test):

    jc_pos_test_pred = nx.jaccard_coefficient(graph, pos_test)
    jc_neg_test_pred = nx.jaccard_coefficient(graph, neg_test)

    jc_pos_score = [p for _, _, p in jc_pos_test_pred]
    jc_neg_score = [n for _, _, n in jc_neg_test_pred]

    jc_all_labels = [1] * len(jc_pos_score) + [0] * len(jc_neg_score)
    jc_all_scores = jc_pos_score + jc_neg_score

    jc_auc = metrics.roc_auc_score(jc_all_labels, jc_all_scores)

    aa_pos_test_pred = nx.resource_allocation_index(graph, pos_test)
    aa_neg_test_pred = nx.resource_allocation_index(graph, neg_test)

    aa_pos_score = [p for _, _, p in aa_pos_test_pred]
    aa_neg_score = [n for _, _, n in aa_neg_test_pred]

    aa_all_labels = [1] * len(aa_pos_score) + [0] * len(aa_neg_score)
    aa_all_scores = aa_pos_score + aa_neg_score

    aa_auc = metrics.roc_auc_score(aa_all_labels, aa_all_scores)

    return jc_auc, aa_auc


def main(params):

    if params.type == 'lfr':
        mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        harmonic_accs = []
        lg_accs = []
        for m in mu:
            jc_aucs = []
            aa_aucs = []
            for r in range(10):
                graph, pos_test_links, neg_test_links = get_lfr_network_data(1000, 3, 1.5, m)
                jc_auc, aa_auc = get_link_pred_auc(graph, pos_test_links, neg_test_links)
                jc_aucs.append(jc_auc)
                aa_aucs.append(aa_auc)
            print(f'{params.dataset} with mu={m} & {np.mean(jc_aucs):.3f} & {np.mean(aa_aucs):.3f}')

    elif params.type == 'node':
        jc_aucs = []
        aa_aucs = []
        for r in range(10):
            graph, pos_test_links, neg_test_links = get_node_label_network_data(params.dataset)
            jc_auc, aa_auc = get_link_pred_auc(graph, pos_test_links, neg_test_links)
            jc_aucs.append(jc_auc)
            aa_aucs.append(aa_auc)
        print(f'{params.dataset} & {np.mean(jc_aucs):.3f} & {np.mean(aa_aucs):.3f}')

    elif params.type == 'real':
        jc_aucs = []
        aa_aucs = []
        for r in range(10):
            graph, pos_test_links, neg_test_links = get_graph(params.dataset)
            jc_auc, aa_auc = get_link_pred_auc(graph, pos_test_links, neg_test_links)
            jc_aucs.append(jc_auc)
            aa_aucs.append(aa_auc)
        print(f'{params.dataset} & {np.mean(jc_aucs):.3f} & {np.mean(aa_aucs):.3f}')

    print(f'Number of nodes in the graph = {len(graph)}')
    print('The components of the graph are as follows -- ')
    print([len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)])

    # fig1 = plt.figure(figsize=(12, 8))
    # plt.plot(harmonic_accs, '-')
    # plt.title('Harmonic function algorithm accuracy as a function of portion of labelled data')
    # plt.xlabel('Portion of labelled data available')
    # plt.ylabel('Accuracy')
    # fig1.savefig(f'{dataset}_hm_acc.png', dpi=fig.dpi)

    # fig2 = plt.figure(figsize=(12, 8))
    # plt.plot(lg_accs, '-')
    # plt.title('Local and global consistency algorithm accuracy as a function of portion of labelled data')
    # plt.xlabel('Portion of labelled data available')
    # plt.ylabel('Accuracy')
    # fig2.savefig(f'{dataset}_lg_acc.png', dpi=fig.dpi)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network data metrics')

    parser.add_argument("--dataset", "-d", type=str, default="football",
                        help="Dataset name")
    parser.add_argument("--type", "-t", type=str, default="real",
                        help="Dataset name")

    params = parser.parse_args()

    main(params)
