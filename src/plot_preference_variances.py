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

def trajectory_rewards(trajectories, device, use_standardizer, stand_reward_table, raw_reward_table):
    # input is shape [N, tMax, 2]
    N = trajectories.shape[0]
    E = stand_reward_table.shape[0]
    rewards = torch.zeros(N, E).to(device)
    for t in range(trajectories.shape[1]):
        states = trajectories[:,t,0]
        actions = trajectories[:,t,1]
        ensemble_indices = torch.arange(E).view(E, 1).expand(E, N)  # shape [E, N]
        state_indices = states.view(1, N).expand(E, N)              # shape [E, N]
        action_indices = actions.view(1, N).expand(E, N)            # shape [E, N]
        if use_standardizer:
            rewards += stand_reward_table[ensemble_indices, state_indices, action_indices, t].T # shape [N, E]
        else:
            rewards += raw_reward_table[ensemble_indices, state_indices, action_indices].T # shape [N, E]
    return rewards.mean(dim=1), rewards



if __name__ == "__main__":

    exp_name = "36_15_varprel_starc"
    seed = 1410#7547#
    epoch = 20

    file_path = f"experiments/{exp_name}/{seed}/raw/full_data_{epoch}.csv"#os.path.join(raw_dir, filename)
    trajectories_df = pd.read_csv(file_path, usecols=["Trajectories1", "Trajectories2"]).astype(int)

    traj_1 = torch.tensor(trajectories_df["Trajectories1"].to_numpy())
    traj_2 = torch.tensor(trajectories_df["Trajectories2"].to_numpy())
    traj_1 = traj_1.reshape((traj_1.shape[0] // 32, 16, 2))
    traj_2 = traj_2.reshape((traj_2.shape[0] // 32, 16, 2))

    sa_frequencies = np.zeros((36, 4))
    for i in range(traj_1.shape[0]):
        for t in range(traj_1.shape[1]):
            sa_frequencies[traj_1[i][t][0]][traj_1[i][t][1]] += 1
            sa_frequencies[traj_2[i][t][0]][traj_2[i][t][1]] += 1
    

    base_config = load_config("base_config")
    config = load_config("36_15_varprel_starc")
    config = override_config(base_config, config)

    device = torch.device('cuda')
    markov_decision_process = LavaPathDetMDP(device)
    reward_standardizer = StarcRewardStandardizer(config['reward_standardizer'], device, markov_decision_process)
    pref_model = EnsemblePreferenceModel(device, reward_standardizer, **config['pref_model'])

    models_path = f"experiments/{exp_name}/{seed}/raw/reward_model_epoch_{epoch}_"
    for i in range(10):
        pref_model.models[i].load_state_dict(torch.load(models_path + f"{i}", weights_only=True))
        pref_model.models[i].eval()

    pref_model.generate_raw_reward_table()

    reward_standardizer.calculate_values(pref_model)
    reward_standardizer.calculate_norms(pref_model)

    pref_model.generate_stand_reward_table()

    raw_reward_table = pref_model.raw_reward_table
    raw_reward_variances_table = raw_reward_table.var(dim=0)
    standardized_reward_table = pref_model.stand_reward_table
    standardized_reward_variances_table = standardized_reward_table.var(dim=0)

    print(pref_model.stand_reward_table.shape)
    print(pref_model.raw_reward_table.shape)



    _, raw_rewards_1 = trajectory_rewards(traj_1, device, False, standardized_reward_table, raw_reward_table)
    _, stand_rewards_1 = trajectory_rewards(traj_1, device, True, standardized_reward_table, raw_reward_table)
    
    _, raw_rewards_2 = trajectory_rewards(traj_2, device, False, standardized_reward_table, raw_reward_table)
    _, stand_rewards_2 = trajectory_rewards(traj_2, device, True, standardized_reward_table, raw_reward_table)


    raw_diff = raw_rewards_1 - raw_rewards_2
    stand_diff = stand_rewards_1 - stand_rewards_2

    raw_pref_probs = torch.sigmoid(raw_diff)
    stand_pref_probs = torch.sigmoid(stand_diff)
    raw_pref_probs_vars = raw_pref_probs.var(dim=1)
    stand_pref_probs_vars = stand_pref_probs.var(dim=1)
    
    test_raw_rew_vars = raw_rewards_1.var(dim=1) + raw_rewards_2.var(dim=1)
    test_stand_rew_vars = stand_rewards_1.var(dim=1) + stand_rewards_2.var(dim=1)



    # raw_reward_table = torch.zeros(36, 4).to(device)
    # raw_reward_variances_table = torch.zeros(36, 4).to(device)
    # standardized_reward_table = torch.zeros(36, 4, 16).to(device)
    # standardized_reward_variances_table = torch.zeros(36, 4, 16).to(device)
    # for s in range(36):
    #     for a in range(4):
    #         sa = torch.tensor([s, a]).to(device).reshape(1, 2)
    #         raw_reward, all_raw_rewards = pref_model.generate_state_actions_reward(sa, None, use_standardizer=False)
    #         raw_reward_table[s][a] = raw_reward
    #         raw_reward_variances_table[s][a] = all_raw_rewards.var()
    #         for t in range(16):
    #             stand_reward, all_stand_rewards = pref_model.generate_state_actions_reward(sa, t, use_standardizer=True)
    #             standardized_reward_table[s][a][t] = stand_reward
    #             standardized_reward_variances_table[s][a][t] = all_stand_rewards.var()
    tables = [raw_reward_table, raw_reward_variances_table, standardized_reward_table, standardized_reward_variances_table]





    # sns.set_theme(style='darkgrid')
    # fig, ax1 = plt.subplots(figsize=(8, 6))
    # ax1.scatter(sa_frequencies.flatten(), raw_reward_variances_table.detach().cpu().numpy().flatten(), color='red')
    # ax1.set_xlabel('Frequency')
    # ax1.set_ylabel('Variance')
    # plt.title('Frequency vs Uncertainty (None)')
    # plt.savefig("freq_vs_unc_none", dpi=300)  # Adjust dpi as needed
    # plt.close()

    # sns.set_theme(style='darkgrid')
    # fig, ax1 = plt.subplots(figsize=(8, 6))
    # ax1.scatter(sa_frequencies.flatten(), standardized_reward_variances_table.detach().cpu().numpy().flatten(), color='red')
    # ax1.set_xlabel('Frequency')
    # ax1.set_ylabel('Variance')
    # plt.title('Frequency vs Uncertainty (Starc)')
    # plt.savefig("freq_vs_unc_starc", dpi=300)  # Adjust dpi as needed
    # plt.close()


    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(raw_pref_probs_vars.detach().cpu().numpy().flatten(), stand_pref_probs_vars.detach().cpu().numpy().flatten(), color='red')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Variance')
    axes[0].set_title('Frequency vs Uncertainty (None)')
    axes[1].scatter(test_stand_rew_vars.detach().cpu().numpy().flatten(), raw_pref_probs_vars.detach().cpu().numpy().flatten(), color='red')
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Frequency vs Uncertainty (Starc)')
    plt.savefig("test", dpi=300)  # Adjust dpi as needed
    plt.close()

        