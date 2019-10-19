import argparse
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np


def plot_results(dataset):

    football_harmonic_accs = np.mean(np.load('football_hm_acc.npy'), axis=1)
    polblogs_harmonic_accs = np.mean(np.load('polblogs_hm_acc.npy'), axis=1)
    polbooks_harmonic_accs = np.mean(np.load('polbooks_hm_acc.npy'), axis=1)
    strike_harmonic_accs = np.mean(np.load('strike_hm_acc.npy'), axis=1)
    karate_harmonic_accs = np.mean(np.load('karate_hm_acc.npy'), axis=1)
    train_portion = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_portion, football_harmonic_accs, '-o')
    plt.plot(train_portion, polblogs_harmonic_accs, '-o')
    plt.plot(train_portion, polbooks_harmonic_accs, '-o')
    plt.plot(train_portion, strike_harmonic_accs, '-o')
    plt.plot(train_portion, karate_harmonic_accs, '-o')
    plt.title('Harmonic function algorithm accuracy as a function of portion of labelled data')
    plt.xlabel('Portion of labelled data available')
    plt.ylabel('Accuracy')
    plt.legend(['football', 'polblogs', 'polbooks', 'strike', 'karate'])
    fig.savefig(f'hm_acc.png', dpi=fig.dpi)

    football_lg_accs = np.mean(np.load('football_lg_acc.npy'), axis=1)
    polblogs_lg_accs = np.mean(np.load('polblogs_lg_acc.npy'), axis=1)
    polbooks_lg_accs = np.mean(np.load('polbooks_lg_acc.npy'), axis=1)
    strike_lg_accs = np.mean(np.load('strike_lg_acc.npy'), axis=1)
    karate_lg_accs = np.mean(np.load('karate_lg_acc.npy'), axis=1)
    train_portion = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_portion, football_lg_accs, '-o')
    plt.plot(train_portion, polblogs_lg_accs, '-o')
    plt.plot(train_portion, polbooks_lg_accs, '-o')
    plt.plot(train_portion, strike_lg_accs, '-o')
    plt.plot(train_portion, karate_lg_accs, '-o')
    plt.title('Local-global consistency algorithm accuracy as a function of portion of labelled data')
    plt.xlabel('Portion of labelled data available')
    plt.ylabel('Accuracy')
    plt.legend(['football', 'polblogs', 'polbooks', 'strike', 'karate'])
    fig.savefig(f'lg_acc.png', dpi=fig.dpi)

    # lg_accs = np.mean(np.load(f'{dataset}_lg_acc.npy'), axis=1)
    # train_portion = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    # fig = plt.figure(figsize=(12, 8))
    # plt.plot(train_portion, lg_accs, '-o')
    # plt.title('Harmonic function algorithm accuracy as a function of portion of labelled data')
    # plt.xlabel('Portion of labelled data available')
    # plt.ylabel('Accuracy')
    # fig.savefig(f'{dataset}_lg_acc.png', dpi=fig.dpi)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network data metrics')

    parser.add_argument("--dataset", "-d", type=str, default="football",
                        help="Dataset name")

    params = parser.parse_args()

    plot_results(params.dataset)
