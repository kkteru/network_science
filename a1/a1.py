import argparse
import os
import pdb
import operator
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import Counter
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import csc_matrix, linalg
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import pearsonr


def get_adj(dataset):

    with open(f'data/{dataset}.edgelist.txt') as f:
        file_data = np.array([list(map(int, line.split())) for line in f.read().split('\n')[:-1]])

    n = np.max(file_data) + 1

    adj = csc_matrix((np.ones(len(file_data), dtype=np.uint8), (file_data[:, 0], file_data[:, 1])), shape=(n, n))

    # remove self loops
    adj.setdiag(0)
    adj.eliminate_zeros()

    # ignore direction
    adj += adj.T

    # remove multi-edges
    nz = adj.nonzero()
    adj_simple = csc_matrix((np.ones(len(nz[0]), dtype=np.uint8), (nz[0], nz[1])), shape=(n, n))

    return adj_simple


def plot_degree_dist(adj, dataset):

    degree_list = np.asarray(adj.sum(0)).squeeze()

    degree_freq = Counter(degree_list)

    if 0 in degree_freq.keys():
        degree_freq.pop(0)

    x = np.log10(list(degree_freq.keys()))
    y = np.log10(list(degree_freq.values()))

    # bins = np.logspace(0, 3, 50)

    # degree_freq_binned = np.histogram(degree_list, bins=bins)

    # idx = np.where(degree_freq_binned[0] != 0)[0]

    coeff = np.polyfit(x, y, deg=1)
    p = np.poly1d(coeff)
    xp = np.log10(np.linspace(1, np.max(list(degree_freq.keys())).max(), 1000))

    print('Slope:', coeff[0])

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'o', xp, p(xp), '-')
    # plt.plot(x, y, 'o')
    plt.title(f'Degree distribution of {dataset} graph (Slope: {coeff[0]})')

    plt.legend(['True degree dist.', 'Line fit'])

    xlocs, _ = plt.xticks()
    ylocs, _ = plt.yticks()
    xlocs_new = np.arange(int(np.max(xlocs)) + 1)
    ylocs_new = np.arange(int(np.max(ylocs)) + 1)
    plt.xticks(xlocs_new, 10**xlocs_new)
    plt.yticks(ylocs_new, 10**ylocs_new)

    # plt.text(2, 1.5, f'Slope : {coeff[0]}', {'color': 'b', 'fontsize': 16})

    fig.savefig(os.path.join(dataset, f'{dataset}_degree_dist.png'), dpi=fig.dpi)


def plot_cc_dist(adj, dataset):

    degree_list = np.asarray(adj.sum(0)).squeeze()
    cc = (adj**3).diagonal() / (degree_list * (degree_list - 1) + 1e-9)

    cc_freq = Counter(cc)
    if 0 in cc_freq.keys():
        cc_freq.pop(0)

    x = list(cc_freq.keys())
    y = list(cc_freq.values())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'o')
    plt.title(f'Clustering coefficient distribution of {dataset} graph (avg: {np.mean(cc)})')

    plt.legend(['True CC distribution'])

    # xlocs, _ = plt.xticks()
    # xlocs_new = np.arange(int(np.max(xlocs)) + 1)
    # plt.xticks(xlocs_new, 10**xlocs_new)
    # ylocs, _ = plt.yticks()
    # ylocs_new = np.arange(int(np.max(ylocs)) + 1)
    # plt.yticks(ylocs_new, 10**ylocs_new)

    # plt.text(2, 1.5, f'Slope : {coeff[0]}', {'color': 'b', 'fontsize': 16})

    fig.savefig(os.path.join(dataset, f'{dataset}_cc_dist.png'), dpi=fig.dpi)


def plot_sp_dist(adj, dataset):
    sp = shortest_path(adj, directed=False, unweighted=True)

    sp_freq = Counter(sp.flatten())
    sp_freq.pop(0)

    x = list(sp_freq.keys())
    y = np.log10(list(sp_freq.values()))

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'o')
    plt.title(f'Shortest path distribution of {dataset} graph (avg: {np.mean(sp)})')

    plt.legend(['True CC distribution'])

    # xlocs, _ = plt.xticks()
    # xlocs_new = np.arange(int(np.max(xlocs)) + 1)
    # plt.xticks(xlocs_new, 10**xlocs_new)
    ylocs, _ = plt.yticks()
    ylocs_new = np.arange(int(np.max(ylocs)) + 1)
    plt.yticks(ylocs_new, 10**ylocs_new)

    # plt.text(2, 1.5, f'Slope : {coeff[0]}', {'color': 'b', 'fontsize': 16})

    fig.savefig(os.path.join(dataset, f'{dataset}_sp_dist.png'), dpi=fig.dpi)


def get_connected_comp(adj, dataset):
    n_components, labels = connected_components(adj, directed=False)

    components_count = Counter(labels)

    with open(os.path.join(dataset, f'{dataset}_cc.txt'), "w+") as f:
        f.write(f'Number of components = {n_components}, number of node in GCC = {max(components_count.values())}')


def plot_eig_dist(adj, dataset):

    degree_list = np.asarray(adj.sum(0)).squeeze()
    n = len(degree_list)
    D = csc_matrix((degree_list, (np.arange(n), np.arange(n))), shape=(n, n), dtype=float)

    laplacian = D - adj

    eig_vals = np.sort(linalg.eigs(laplacian, k=20, which='LM', return_eigenvectors=False))

    bins = np.logspace(0, 3, 50)

    eig_vals_freq = np.histogram(eig_vals, bins=bins)

    fig = plt.figure(figsize=(12, 8))
    # plt.bar(bins[:-1], eig_vals_freq[0], widths)
    plt.plot(bins[:-1], eig_vals_freq[0], 'o')
    plt.xscale('log')
    plt.title(f'Eigenvalue distribution of {dataset} graph (spectral gap: {eig_vals[-1] - eig_vals[-2]})')
    plt.legend(['True eigenvalue distribution'])

    print(f'First eigenvalues : {eig_vals[0]}')

    # xlocs, _ = plt.xticks()
    # xlocs_new = np.arange(int(np.max(xlocs)) + 1)
    # plt.xticks(xlocs_new, 10**xlocs_new)

    # xlocs, _ = plt.xticks()
    # xlocs_new = np.arange(int(np.max(xlocs)) + 1)
    # plt.xticks(xlocs_new, 10**xlocs_new)
    # ylocs, _ = plt.yticks()
    # ylocs_new = np.arange(int(np.max(ylocs)) + 1)
    # plt.yticks(ylocs_new, 10**ylocs_new)

    # plt.text(2, 1.5, f'Slope : {coeff[0]}', {'color': 'b', 'fontsize': 16})

    fig.savefig(os.path.join(dataset, f'{dataset}_eig_dist.png'), dpi=fig.dpi)


def plot_degree_corr(adj, dataset):
    degree_list = np.asarray(adj.sum(0)).squeeze()

    nz = adj.nonzero()

    deg_corr = csc_matrix((np.ones(len(nz[0]), dtype=np.uint8), (degree_list[nz[0]], degree_list[nz[1]])), shape=(int(degree_list.max() + 1), int(degree_list.max() + 1)))

    fig = plt.figure(figsize=(12, 8))
    plt.plot(degree_list[nz[0]], degree_list[nz[1]], 'o', alpha=0.1)
    plt.title(f'Degree correlation of {dataset} graph (overall correlation: {pearsonr(degree_list[nz[0]], degree_list[nz[1]])[0]})')

    fig.savefig(os.path.join(dataset, f'{dataset}_degree_corr.png'), dpi=fig.dpi)


def plot_degree_cc_corr(adj, dataset):
    degree_list = np.asarray(adj.sum(0)).squeeze()
    cc_list = (adj**3).diagonal() / (degree_list * (degree_list - 1) + 1e-9)

    nz = adj.nonzero()
    deg_cc_corr = csc_matrix((np.ones(len(nz[0]), dtype=np.uint8), (degree_list[nz[0]], cc_list[nz[1]])), shape=(int(degree_list.max() + 1), int(cc_list.max() + 1)))

    fig = plt.figure(figsize=(12, 8))
    plt.plot(degree_list[nz[0]], cc_list[nz[1]], 'o', alpha=0.1)
    plt.title(f'Degree-cc correlation of {dataset} graph')

    fig.savefig(os.path.join(dataset, f'{dataset}_degree_cc_corr.png'), dpi=fig.dpi)


def get_ba_adj(num_nodes, num_links):

    init_nodes = int((num_nodes + 1 - np.sqrt((num_nodes + 1)**2 - 4 * num_links)) / 2 + 1)
    init_links = init_nodes

    num_links_per_iter = int((num_links - init_links) / (num_nodes - init_nodes))

    print(init_nodes, init_links, num_links_per_iter)

    # initial conditions
    # TODO : change the initial conditions
    nodes_list = list(np.arange(num_nodes))
    source_nodes = list(np.arange(init_nodes))
    target_nodes = list(np.arange(init_nodes))[1:] + list(np.arange(init_nodes))[:1]
    # target_nodes = np.random.choice(nodes_list, size=init_nodes, replace=True).tolist()
    nodes_degree = np.array([0] * num_nodes)
    nodes_degree[:init_nodes] = 2

    # build the graph
    for t in tqdm(range(init_nodes, num_nodes)):
        source_nodes += np.random.choice(nodes_list[t], size=num_links_per_iter, replace=False, p=(nodes_degree / np.sum(nodes_degree))[:t]).tolist()
        target_nodes += [t] * num_links_per_iter

        nodes_degree[source_nodes] += 1
        nodes_degree[t] = num_links_per_iter

    # create the adjacency matrix
    adj = csc_matrix((np.ones(len(source_nodes), dtype=np.uint8), (source_nodes, target_nodes)), shape=(num_nodes, num_nodes))

    # remove self loops
    adj.setdiag(0)
    adj.eliminate_zeros()

    # ignore direction
    adj += adj.T

    # remove multi-edges
    nz = adj.nonzero()
    adj_simple = csc_matrix((np.ones(len(nz[0]), dtype=np.uint8), (nz[0], nz[1])), shape=(num_nodes, num_nodes))

    return adj_simple


def get_ba_adj1(num_nodes, num_links):

    init_nodes = int((num_nodes + 1 - np.sqrt((num_nodes + 1)**2 - 4 * num_links)) / 2 + 1)
    init_links = init_nodes

    num_links_per_iter = int((num_links - init_links) / (num_nodes - init_nodes))

    print(init_nodes, init_links, num_links_per_iter)

    # initial conditions
    # TODO : change the initial conditions
    nodes_list = list(np.arange(num_nodes))
    source_nodes = np.arange(init_nodes).tolist()
    target_nodes = list(np.arange(init_nodes))[1:] + list(np.arange(init_nodes))[:1]
    # target_nodes = np.random.choice(nodes_list, size=init_nodes, replace=True).tolist()
    # nodes_degree = np.array([0] * num_nodes)
    # nodes_degree[:init_nodes] = 2

    # build the graph
    for t in tqdm(range(init_nodes, num_nodes)):
        nodes = np.random.randint(2 * len(source_nodes), size=num_links_per_iter).tolist()

        new_source_nodes = []
        for node in nodes:
            if node < len(source_nodes):
                new_source_nodes.append(source_nodes[node])
            else:
                new_source_nodes.append(target_nodes[node - len(source_nodes)])

        source_nodes += new_source_nodes
        target_nodes += [t] * num_links_per_iter

        # nodes_degree[source_nodes] += 1
        # nodes_degree[t] = num_links_per_iter

    # create the adjacency matrix
    adj = csc_matrix((np.ones(len(source_nodes), dtype=np.uint8), (source_nodes, target_nodes)), shape=(num_nodes, num_nodes))

    # remove self loops
    adj.setdiag(0)
    adj.eliminate_zeros()

    # ignore direction
    adj += adj.T

    # remove multi-edges
    nz = adj.nonzero()
    adj_simple = csc_matrix((np.ones(len(nz[0]), dtype=np.uint8), (nz[0], nz[1])), shape=(num_nodes, num_nodes))

    return adj_simple


def main(params):

    print(f'Computing stats for {params.dataset}')
    print('========================')

    adj = get_adj(params.dataset)

    tot_nodes = adj.shape[0]
    tot_links = adj.sum()

    print(f'Total nodes = {tot_nodes}, total links = {tot_links}')

    plot_degree_dist(adj, params.dataset)
    plot_degree_corr(adj, params.dataset)

    if adj.shape[0] > 5000:
        sampled_nodes = np.random.choice(np.arange(adj.shape[0]), size=5000, replace=False)
        adj = adj[sampled_nodes, :][:, sampled_nodes]

        print(f'Sampled nodes = {adj.shape[0]}, Sampled links = {adj.sum()}')

    plot_cc_dist(adj, params.dataset)
    plot_degree_cc_corr(adj, params.dataset)
    plot_sp_dist(adj, params.dataset)
    get_connected_comp(adj, params.dataset)

    plot_eig_dist(adj, params.dataset)

    ####################
    print(f'Computing stats for {params.dataset}_ba')
    print('========================')

    ba_adj = get_ba_adj1(num_nodes=tot_nodes, num_links=tot_links // 2)

    print(f'Total nodes in BA model = {ba_adj.shape[0]}, total links in BA model = {ba_adj.sum()}')

    plot_degree_dist(ba_adj, params.dataset + '_ba')
    plot_degree_corr(ba_adj, params.dataset + '_ba')

    if ba_adj.shape[0] > 5000:
        sampled_nodes = np.random.choice(np.arange(ba_adj.shape[0]), size=5000, replace=False)
        ba_adj = ba_adj[sampled_nodes, :][:, sampled_nodes]

        print(f'Sampled nodes = {ba_adj.shape[0]}, Sampled links = {ba_adj.sum()}')

    plot_cc_dist(ba_adj, params.dataset + '_ba')
    plot_degree_cc_corr(ba_adj, params.dataset + '_ba')
    plot_sp_dist(ba_adj, params.dataset + '_ba')
    get_connected_comp(ba_adj, params.dataset + '_ba')

    plot_eig_dist(adj, params.dataset + '_ba')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Network data metrics')

    parser.add_argument("--dataset", "-d", type=str, default="metabolic",
                        help="Dataset name")

    params = parser.parse_args()

    if not os.path.exists(params.dataset):
        os.makedirs(params.dataset)
    if not os.path.exists(params.dataset + '_ba'):
        os.makedirs(params.dataset + '_ba')

    main(params)
