from trajectory_datasets import *
from markov_desicion_processes import *
from data_selection_policies import *
from trainers import *
from preference_models import *
from policies import *
from reward_standardizers import *
from configs import load_config, override_config
import utils
import argparse
import seaborn as sns
from modules import MCDropoutRewardMLP

import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def build_data_stats(trajectories1, trajectories2):
    hist_s_1, bins_s_1 = np.histogram(torch.flatten(trajectories1[:,:,0]).cpu().numpy(), bins=36, range=(0,36))
    hist_s_2, bins_s_2 = np.histogram(torch.flatten(trajectories2[:,:,0]).cpu().numpy(), bins=36, range=(0,36))
    hist_a_1, bins_a_1 = np.histogram(torch.flatten(trajectories1[:,:,1]).cpu().numpy(), bins=4, range=(0,4))
    hist_a_2, bins_a_2 = np.histogram(torch.flatten(trajectories1[:,:,1]).cpu().numpy(), bins=4, range=(0,4))
    return {
        'states': (hist_s_1+hist_s_2, bins_s_1),
        'actions': (hist_a_1+hist_a_2, bins_a_1)
    }

def plot_dataset_histogram(hist_states, bins_states, hist_actions, bins_actions):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    values_states = []
    for count, bin_start, bin_end in zip(hist_states, bins_states[:-1], bins_states[1:]):
        values_states.extend([bin_start] * count)

    values_actions = []
    for count, bin_start, bin_end in zip(hist_actions, bins_actions[:-1], bins_actions[1:]):
        values_actions.extend([bin_start] * count)

    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot the first histogram
    sns.histplot(values_states, bins=bins_states, kde=False, ax=axes[0])
    axes[0].set_title('States Distribution')
    axes[0].set_xlabel('States Value')
    axes[0].set_ylabel('Frequency')

    sns.heatmap(hist_states.reshape(6,6), ax=axes[1], annot=True, annot_kws={'size': 10}, fmt='g')
    axes[1].set_title('States Distribution')

    # Plot the second histogram
    sns.histplot(values_actions, bins=bins_actions, kde=False, ax=axes[2])
    axes[2].set_title('Actions Distribution')
    axes[2].set_xlabel('Action Value')
    axes[2].set_ylabel('Frequency')

    plt.savefig(f"dataset_histogram", dpi=300)  # Adjust dpi as needed
    plt.close()

def plot_state_distributions(hist_states_train, hist_states_test):
    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.heatmap(hist_states_train.reshape(6,6), ax=axes[0], annot=True, annot_kws={'size': 10}, fmt='g', yticklabels=False, xticklabels=False, linewidths=1, linecolor='black', vmin=0, cbar=False)
    axes[0].set_title('State Frequency (Pool Dataset)')

    sns.heatmap(hist_states_test.reshape(6,6), ax=axes[1], annot=True, annot_kws={'size': 10}, fmt='g', yticklabels=False, xticklabels=False, linewidths=1, linecolor='black', vmin=0, cbar=False)
    axes[1].set_title('State Frequency (Test Dataset)')

    plt.savefig(f"dataset_state_distributions", dpi=300)  # Adjust dpi as needed
    plt.close()


if __name__ == "__main__":

    device = torch.device('cuda')
    markov_decision_process = LavaPathNonDetMDP(None, device)
    # dataset = Random36NonDetTrajectoryDataset(None, markov_decision_process, device)
    # dataset.generate_dataset(15000)
    # trajectories1, trajectories2 = dataset.samples

    # data_stats = build_data_stats(trajectories1, trajectories2)
    # hist_states, bins_states = data_stats['states']
    # hist_actions, bins_actions = data_stats['actions']
    # plot_dataset_histogram(hist_states, bins_states, hist_actions, bins_actions)
    # print(hist_states)

    dataset = RandomNonDetTrajectoryDataset(None, markov_decision_process, device)
    dataset.generate_dataset(15000)
    trajectories1, trajectories2 = dataset.samples
    data_stats = build_data_stats(trajectories1, trajectories2)
    hist_states_train, _ = data_stats['states']

    dataset = Random36NonDetTrajectoryDataset(None, markov_decision_process, device)
    dataset.generate_dataset(15000)
    trajectories1, trajectories2 = dataset.samples
    data_stats = build_data_stats(trajectories1, trajectories2)
    hist_states_test, _ = data_stats['states']

    plot_state_distributions(hist_states_train, hist_states_test)

        