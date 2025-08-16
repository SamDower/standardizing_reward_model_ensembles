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
import pandas as pd

import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def load_trajectories(exp_name, seed, epoch):
    file_path = f"experiments/{exp_name}/{seed}/raw/full_data_{epoch}.csv"
    trajectories_df = pd.read_csv(file_path, usecols=["Trajectories1", "Trajectories2"]).astype(int)

    traj_1 = trajectories_df["Trajectories1"].to_numpy()
    traj_2 = trajectories_df["Trajectories2"].to_numpy()
    traj_1 = traj_1.reshape((traj_1.shape[0] // 32, 16, 2))
    traj_2 = traj_2.reshape((traj_2.shape[0] // 32, 16, 2))
    return traj_1, traj_2

def calculate_sa_frequencies(traj_1, traj_2):
    sa_frequencies = np.zeros((36, 4))
    for i in range(traj_1.shape[0]):
        for t in range(traj_1.shape[1]):
            sa_frequencies[traj_1[i][t][0]][traj_1[i][t][1]] += 1
            sa_frequencies[traj_2[i][t][0]][traj_2[i][t][1]] += 1
    return sa_frequencies

def generate_reward_variance_tables(exp_name, seed, epoch):
    base_config = load_config("base_config")
    config = load_config(f"{exp_name}")
    config = override_config(base_config, config)

    device = torch.device('cuda')
    markov_decision_process = LavaPathDetMDP(config['mdp'], device)
    reward_standardizer = StarcRewardStandardizer(config['reward_standardizer'], device, markov_decision_process)
    pref_model = EnsemblePreferenceModel(device, reward_standardizer, **config['pref_model'])

    models_path = f"experiments/{exp_name}/{seed}/raw/reward_model_epoch_{epoch}_"
    for i in range(10):
        pref_model.models[i].load_state_dict(torch.load(models_path + f"{i}", weights_only=True))
        pref_model.models[i].eval()
    
    pref_model.generate_raw_reward_table()

    reward_standardizer.calculate_values(pref_model)
    reward_standardizer.calculate_norms(pref_model)

    raw_reward_table = torch.zeros(36, 4).to(device)
    raw_reward_variances_table = torch.zeros(36, 4).to(device)
    standardized_reward_table = torch.zeros(36, 4, 16).to(device)
    standardized_reward_variances_table = torch.zeros(36, 4, 16).to(device)
    for s in range(36):
        for a in range(4):
            sa = torch.tensor([s, a]).to(device).reshape(1, 2)
            raw_reward, all_raw_rewards = pref_model.generate_state_actions_reward(sa, None, use_standardizer=False)
            raw_reward_table[s][a] = raw_reward
            raw_reward_variances_table[s][a] = all_raw_rewards.var()
            for t in range(16):
                stand_reward, all_stand_rewards = pref_model.generate_state_actions_reward(sa, t, use_standardizer=True)
                standardized_reward_table[s][a][t] = stand_reward
                standardized_reward_variances_table[s][a][t] = all_stand_rewards.var()
    return raw_reward_variances_table, standardized_reward_variances_table


if __name__ == "__main__":

    exp_name = "36_15_varprel_starc"
    seed = 1560
    epoch = 49

    # exp_name = "36_15_varp_none"
    # seed = 2436
    # epoch = 10

    traj_1, traj_2 = load_trajectories(exp_name, seed, epoch)

    sa_frequencies = calculate_sa_frequencies(traj_1, traj_2)
    raw_reward_variances_table, standardized_reward_variances_table = generate_reward_variance_tables(exp_name, seed, epoch)


    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].scatter(sa_frequencies.flatten(), raw_reward_variances_table.detach().cpu().numpy().flatten(), color='red', alpha=1)
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Uncertainty')
    axes[0].set_title('Frequency vs Uncertainty (No Standardizing)')
    axes[1].scatter(sa_frequencies.flatten(), standardized_reward_variances_table[:,:,5].detach().cpu().numpy().flatten(), color='red', alpha=1)
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('Uncertainty')
    axes[1].set_title('Frequency vs Uncertainty (With Standardizing)')
    plt.savefig("freq_vs_unc", dpi=300)  # Adjust dpi as needed
    plt.close()










































# def reward_simulator_version():

#     exp_name = "rew36_15_varr_starc"
#     seed = 2311#7547#
#     epoch = 20

#     file_path = f"experiments/{exp_name}/{seed}/raw/full_data_{epoch}.csv"#os.path.join(raw_dir, filename)
#     #trajectories_df = pd.read_csv(file_path, usecols=["Trajectories1", "Trajectories2"]).astype(int)
#     trajectories_df = pd.read_csv(file_path, usecols=["Trajectories"])
#     print(trajectories_df)#.astype(int)
#     print(trajectories_df.dtypes)

#     # traj_1 = trajectories_df["Trajectories1"].to_numpy()
#     # traj_2 = trajectories_df["Trajectories2"].to_numpy()
#     traj = trajectories_df["Trajectories"].to_numpy()
#     # traj_1 = traj_1.reshape((traj_1.shape[0] // 32, 16, 2))
#     # traj_2 = traj_2.reshape((traj_2.shape[0] // 32, 16, 2))
#     traj = traj.reshape((traj.shape[0] // 32, 16, 2))

#     sa_frequencies = np.zeros((36, 4))
#     for i in range(traj.shape[0]):
#         for t in range(traj.shape[1]):
#             sa_frequencies[traj[i][t][0]][traj[i][t][1]] += 1
#             #sa_frequencies[traj_2[i][t][0]][traj_2[i][t][1]] += 1
    
#     print(sa_frequencies)



#     base_config = load_config("base_config")
#     config = load_config(f"{exp_name}")
#     config = override_config(base_config, config)

#     device = torch.device('cuda')
#     markov_decision_process = LavaPathDetMDP(config['mdp'], device)
#     reward_standardizer = StarcRewardStandardizer(config['reward_standardizer'], device, markov_decision_process)
#     pref_model = EnsemblePreferenceModel(device, reward_standardizer, **config['pref_model'])

#     models_path = f"experiments/{exp_name}/{seed}/raw/reward_model_epoch_{epoch}_"
#     for i in range(10):
#         pref_model.models[i].load_state_dict(torch.load(models_path + f"{i}", weights_only=True))
#         pref_model.models[i].eval()

#     reward_standardizer.calculate_values(pref_model)
#     reward_standardizer.calculate_norms(pref_model)

#     #pref_model.generate_stand_reward_table()

#     raw_reward_table = torch.zeros(36, 4).to(device)
#     raw_reward_variances_table = torch.zeros(36, 4).to(device)
#     standardized_reward_table = torch.zeros(36, 4, 16).to(device)
#     standardized_reward_variances_table = torch.zeros(36, 4, 16).to(device)
#     for s in range(36):
#         for a in range(4):
#             sa = torch.tensor([s, a]).to(device).reshape(1, 2)
#             raw_reward, all_raw_rewards = pref_model.generate_state_actions_reward(sa, None, use_standardizer=False)
#             raw_reward_table[s][a] = raw_reward
#             raw_reward_variances_table[s][a] = all_raw_rewards.var()
#             for t in range(16):
#                 stand_reward, all_stand_rewards = pref_model.generate_state_actions_reward(sa, t, use_standardizer=True)
#                 standardized_reward_table[s][a][t] = stand_reward
#                 standardized_reward_variances_table[s][a][t] = all_stand_rewards.var()
#     tables = [raw_reward_table, raw_reward_variances_table, standardized_reward_table, standardized_reward_variances_table]





#     sns.set_theme(style='darkgrid')
#     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#     axes[0].scatter(sa_frequencies.flatten(), raw_reward_variances_table.detach().cpu().numpy().flatten(), color='red')
#     axes[0].set_xlabel('Frequency')
#     axes[0].set_ylabel('Variance')
#     axes[0].set_title('Frequency vs Uncertainty (None)')
#     axes[1].scatter(sa_frequencies.flatten(), standardized_reward_variances_table[:,:,0].detach().cpu().numpy().flatten(), color='red')
#     axes[1].set_xlabel('Frequency')
#     axes[1].set_ylabel('Variance')
#     axes[1].set_title('Frequency vs Uncertainty (Starc)')
#     plt.savefig("freq_vs_unc", dpi=300)  # Adjust dpi as needed
#     plt.close()