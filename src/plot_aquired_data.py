from trajectory_datasets import *
from markov_desicion_processes import *
from data_selection_policies import *
from trainers import *
from preference_models import *
from policies import *
from reward_standardizers import *
from preference_simulators import PreferenceOptimizationSimulator
from configs import load_config, override_config
import utils
import argparse
import seaborn as sns
from modules import MCDropoutRewardMLP
import pandas as pd

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

def plot_sa_frequencies(sa_frequencies):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for a in range(4):
        heatmap = sa_frequencies[:,a]
        sns.heatmap(heatmap.reshape(6,6), ax=axes[a], annot=True, annot_kws={'size': 10}, fmt='g')
        axes[a].set_title(f'a={a}')
    plt.savefig(f"sa_frequencies", dpi=300)  # Adjust dpi as needed
    plt.close()




if __name__ == "__main__":

    exp_name = "varp_random_none_new"
    seed = 5783#7547#
    epoch = 23

    file_path = f"experiments/{exp_name}/{seed}/raw/full_data_{epoch}.csv"#os.path.join(raw_dir, filename)
    trajectories_df = pd.read_csv(file_path, usecols=["Trajectories1", "Trajectories2"]).astype(int)

    trajectories1 = trajectories_df["Trajectories1"].to_numpy()
    trajectories2 = trajectories_df["Trajectories2"].to_numpy()
    trajectories1 = torch.from_numpy(trajectories1.reshape((trajectories1.shape[0] // 48, 16, 3)))
    trajectories2 = torch.from_numpy(trajectories2.reshape((trajectories2.shape[0] // 48, 16, 3)))

    sa_frequencies = np.zeros((36, 4))
    for i in range(trajectories1.shape[0]):
        for t in range(trajectories1.shape[1]):
            sa_frequencies[trajectories1[i][t][0]][trajectories1[i][t][1]] += 1
            sa_frequencies[trajectories2[i][t][0]][trajectories2[i][t][1]] += 1
    
    plot_sa_frequencies(sa_frequencies)

    for i in range(trajectories1.shape[0]):
        if i % 15 == 0:
            print("============== New Batch ==============")
        
        print(f"{trajectories1[i][:,:2].flatten()}")
        print(f"{trajectories2[i][:,:2].flatten()}")

    
    # data_stats = build_data_stats(trajectories1, trajectories2)
    # hist_states, bins_states = data_stats['states']
    # hist_actions, bins_actions = data_stats['actions']
    # plot_dataset_histogram(hist_states, bins_states, hist_actions, bins_actions)
    # print(hist_states)

        