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
    node_comm = np.array([graph.nodes[v]['value'] for v in graph])

    largest_cc = max(nx.connected_components(graph), key=len)

    graph = graph.subgraph(largest_cc)

    return nx.Graph(graph)


def get_node_label_network_data(dataset):

    adj_list = np.load(f'./ind.{dataset}.graph')
    train_labels = np.load(f'./ind.{dataset}.y')
    test_labels = np.load(f'./ind.{dataset}.ty')

    graph = nx.from_dict_of_lists(adj_list)

    return graph


def get_lfr_network_data(n, tau1, tau2, mu):
    graph = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=30, seed=10)

    comm = list({frozenset(graph.nodes[v]['community']) for v in graph})

    node_comm = np.empty(n)

    for i, c in enumerate(comm):
        for node in c:
            node_comm[node] = i

    return graph, node_comm


def classify_real_graph_nodes(graph, train_portion=0.8):
    all_labels = nx.get_node_attributes(graph, 'value')

    nodes = np.array(list(all_labels.keys()))
    nodes_idx = np.arange(len(nodes))
    np.random.shuffle(nodes_idx)

    train_nodes_idx = nodes_idx[:int(train_portion * len(nodes))]
    test_nodes_idx = nodes_idx[int(train_portion * len(nodes)):]

    train_labels = {train_node: all_labels[train_node] for train_node in nodes[train_nodes_idx]}
    test_labels = {test_node: all_labels[test_node] for test_node in nodes[test_nodes_idx]}

    nx.set_node_attributes(graph, train_labels, 'train_labels')

    harmonic_pred = node_classification.harmonic_function(graph, label_name='train_labels')
    lg_pred = node_classification.local_and_global_consistency(graph, label_name='train_labels')

    harmonic_acc = metrics.accuracy_score(np.array(list(test_labels.values())), np.array(harmonic_pred)[test_nodes_idx])
    lg_acc = metrics.accuracy_score(np.array(list(test_labels.values())), np.array(lg_pred)[test_nodes_idx])

    return harmonic_acc, lg_acc


def classify_gcn_data(graph, dataset):
    train_y = np.argmax(np.load(f'./ind.{dataset}.y'), axis=1)
    test_y = np.argmax(np.load(f'./ind.{dataset}.ty'), axis=1)

    test_idx = parse_index_file(f'./ind.{dataset}.test.index')

    train_labels = {train_node: train_y[train_node] for train_node in range(len(train_y))}
    test_labels = {test_node: test_y[test_node] for test_node in range(len(test_y))}

    nx.set_node_attributes(graph, train_labels, 'train_labels')

    harmonic_pred = node_classification.harmonic_function(graph, label_name='train_labels')
    lg_pred = node_classification.local_and_global_consistency(graph, label_name='train_labels')

    harmonic_acc = metrics.accuracy_score(test_y, np.array(harmonic_pred)[test_idx])
    lg_acc = metrics.accuracy_score(test_y, np.array(lg_pred)[test_idx])

    return harmonic_acc, lg_acc


def classifiy_lfr(graph, node_labels, train_portion=0.8):

    nodes_idx = np.arange(len(node_labels))
    np.random.shuffle(nodes_idx)

    train_nodes_idx = nodes_idx[:int(train_portion * len(node_labels))]
    test_nodes_idx = nodes_idx[int(train_portion * len(node_labels)):]

    train_labels = {train_node: node_labels[train_node] for train_node in train_nodes_idx}

    nx.set_node_attributes(graph, train_labels, 'train_labels')

    harmonic_pred = node_classification.harmonic_function(graph, label_name='train_labels')
    lg_pred = node_classification.local_and_global_consistency(graph, label_name='train_labels')

    harmonic_acc = metrics.accuracy_score(node_labels[test_nodes_idx], np.array(harmonic_pred)[test_nodes_idx])
    lg_acc = metrics.accuracy_score(node_labels[test_nodes_idx], np.array(lg_pred)[test_nodes_idx])

    return harmonic_acc, lg_acc


def main(params):

    if params.type == 'lfr':
        mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        harmonic_accs = []
        lg_accs = []
        for m in mu:
            hm = []
            lg = []
            for r in range(10):
                graph, comm = get_lfr_network_data(1000, 3, 1.5, m)

                harmonic_acc, lg_acc = classifiy_lfr(graph, comm)

                hm.append(harmonic_acc)
                lg.append(lg_acc)

            print(hm.mean(), lg.mean())

            harmonic_accs.append(hm.mean())
            lg_accs.append(lg.mean())

        np.save('lfr_hm_acc.npy', np.array(harmonic_accs))
        np.save('lfr_lg_acc.npy', np.array(lg_accs))
        print('Mean performance: ', np.mean(harmonic_accs), np.mean(lg_accs))

    elif params.type == 'node':
        graph = get_node_label_network_data(params.dataset)

        print(classify_gcn_data(graph, params.dataset))

    elif params.type == 'real':
        graph = get_graph(params.dataset)

        train_portions = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

        harmonic_accs = []
        lg_accs = []

        for tp in train_portions:
            h = []
            a = []
            for run in range(10):
                harmonic_acc, lg_acc = classify_real_graph_nodes(graph, train_portion=tp)
                print(tp, harmonic_acc, lg_acc)
                h.append(harmonic_acc)
                a.append(lg_acc)
            harmonic_accs.append(h)
            lg_accs.append(a)

        np.save(f'{params.dataset}_hm_acc.npy', np.array(harmonic_accs))
        np.save(f'{params.dataset}_lg_acc.npy', np.array(lg_accs))

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
