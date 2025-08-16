from data_selection_policies import *
from trainers import *
from preference_models import *
from policies import *
from markov_desicion_processes import * 
from trajectory_datasets import * 
from reward_standardizers import *
from preference_simulators import PreferenceOptimizationSimulator
from configs import load_config, override_config
import utils
import argparse
import seaborn as sns
from modules import MCDropoutRewardMLP

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_reward_tables(reward_tables, filepath):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    # Create the subplots
    fig, axes = plt.subplots(len(reward_tables), 4, figsize=(24, 24))

    for table_index in range(len(reward_tables)):
        table = reward_tables[table_index].detach().cpu()
        for a in range(4):
            sub_table = table[:,a].reshape(6,6)
            sns.heatmap(sub_table, vmin=torch.min(table), vmax=torch.max(table), ax=axes[table_index,a])
            axes[table_index,a].set_title(f'R(s,{a})')
    
    plt.savefig(f"{filepath}", dpi=300)  # Adjust dpi as needed
    plt.close()

if __name__ == "__main__":

    base_config = load_config("base_config")
    config = load_config("varp_random_starc")
    config = override_config(base_config, config)

    device = torch.device('cuda')
    markov_decision_process = LavaPathDetMDP(device)
    reward_standardizer = StarcRewardStandardizer(config['reward_standardizer'], device, markov_decision_process)
    pref_model = EnsemblePreferenceModel(device, reward_standardizer, **config['pref_model'])

    models_path = "experiments/varp_random_none/1425/raw/reward_model_epoch_47_"
    for i in range(10):
        pref_model.models[i].load_state_dict(torch.load(models_path + f"{i}", weights_only=True))
        pref_model.models[i].eval()

    reward_standardizer.train_policies(pref_model)
    reward_standardizer.calculate_norms(pref_model)


    # per_model_reward_tables = [torch.zeros(36, 4).to(device), torch.zeros(36, 4).to(device), torch.zeros(36, 4).to(device), torch.zeros(36, 4).to(device)]
    # for s in range(36):
    #     for a in range(4):
    #         sa = torch.tensor([s, a]).to(device).reshape(1, 2)
    #         mean_reward, all_rewards = pref_model.generate_state_actions_reward(sa, use_standardizer=False)
    #         for i in range(4):
    #             per_model_reward_tables[i][s][a] = all_rewards.reshape(10)[i]
    # plot_reward_tables(per_model_reward_tables, "per_model_reward_tables")

    # raw_reward_table = torch.zeros(36, 4).to(device)
    # raw_reward_variances_table = torch.zeros(36, 4).to(device)
    # standardized_reward_table = torch.zeros(36, 4).to(device)
    # standardized_reward_variances_table = torch.zeros(36, 4).to(device)
    # for s in range(36):
    #     for a in range(4):
    #         sa = torch.tensor([s, a]).to(device).reshape(1, 2)
    #         raw_reward, all_raw_rewards = pref_model.generate_state_actions_reward(sa, use_standardizer=False)
    #         raw_reward_table[s][a] = raw_reward
    #         raw_reward_variances_table[s][a] = all_raw_rewards.var()

    #         stand_reward, all_stand_rewards = pref_model.generate_state_actions_reward(sa, use_standardizer=True)
    #         standardized_reward_table[s][a] = stand_reward
    #         standardized_reward_variances_table[s][a] = all_stand_rewards.var()
    # tables = [raw_reward_table, raw_reward_variances_table, standardized_reward_table, standardized_reward_variances_table]
    # plot_reward_tables(tables, "ensemble_reward_tables")




    dataset = RandomTrajectoryDataset(config, markov_decision_process, device)
    dataset.generate_dataset(500)
    trajectories1, trajectories2 = dataset.samples

    raw_reward1, all_raw_rewards1 = pref_model.generate_reward(trajectories1, use_standardizer=False)
    raw_reward2, all_raw_rewards2 = pref_model.generate_reward(trajectories2, use_standardizer=False)
    stand_reward1, all_stand_rewards1 = pref_model.generate_reward(trajectories1, use_standardizer=True)
    stand_reward2, all_stand_rewards2 = pref_model.generate_reward(trajectories2, use_standardizer=True)

    raw_differences = (raw_reward1 - raw_reward2).flatten().detach().cpu().numpy()
    stand_differences = (stand_reward1 - stand_reward2).flatten().detach().cpu().numpy()

    sns.set_theme(style='darkgrid')
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(raw_differences, stand_differences, color='blue')
    ax1.set_xlabel('$R(\tau_1) - R(\tau_2)$')
    ax1.set_ylabel('$R^s(\tau_1) - R^s(\tau_2)$')
    plt.title('Differences in rewards')
    plt.savefig("reward_differences", dpi=300)  # Adjust dpi as needed
    plt.close()




    # dataset = RandomTrajectoryDataset(config, markov_decision_process, device)
    # dataset.generate_dataset(500)
    # trajectories, _ = dataset.samples

    # raw_reward, all_raw_rewards = pref_model.generate_reward(trajectories, use_standardizer=False)
    # raw_vars = all_raw_rewards.var(axis=1).flatten().detach().cpu().numpy()
    # raw_vars_from_sum = pref_model.generate_sum_of_variances(trajectories, use_standardizer=False).flatten().detach().cpu().numpy()

    # stand_reward, all_stand_rewards = pref_model.generate_reward(trajectories, use_standardizer=True)
    # stand_vars = all_stand_rewards.var(axis=1).flatten().detach().cpu().numpy()
    # stand_vars_from_sum = pref_model.generate_sum_of_variances(trajectories, use_standardizer=True).flatten().detach().cpu().numpy()


    # sns.set_theme(style='darkgrid')
    # fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    # axes[0].scatter(raw_vars, raw_vars_from_sum, color='green')
    # axes[0].set_xlabel('Variances')
    # axes[0].set_ylabel('Sum of variances')
    # axes[1].scatter(stand_vars, stand_vars_from_sum, color='green')
    # axes[1].set_xlabel('Variances')
    # axes[1].set_ylabel('Sum of variances')
    # plt.title('Variances vs sum of variances')
    # plt.savefig("variance_sums", dpi=300)  # Adjust dpi as needed
    # plt.close()

    















    # starc_standardizer = StarcBanditRewardStandardizerForPlots(device)
    # policy = UniformPolicy(config['policy'], device)


    # partial_prompt_action_pairs = []
    # partial_inputs = []
    # for i in range(5):
    #     prompts = torch.ones(100).to(device) * i * 0.25
    #     actions = torch.linspace(0, 1, steps=100).to(device)
    #     partial_prompt_action_pairs.append((prompts, actions))
        
    #     inputs = _generate_reward_model_inputs(partial_prompt_action_pairs[i], features=True)
    #     inputs, _, _ = _standardize_features(inputs, 0, 1)
    #     partial_inputs.append(inputs)

    
    # linspace_50 = torch.linspace(0, 1, steps=50).to(device)
    # full_prompts = linspace_50.repeat_interleave(50)
    # full_actions = linspace_50.tile(50)
    # full_prompt_action_pairs = (full_prompts, full_actions)
    
    # full_inputs = _generate_reward_model_inputs(full_prompt_action_pairs, features=True)
    # full_inputs, _, _ = _standardize_features(full_inputs, 0, 1)

    # full_raw_rewards_array = []
    # full_starc_rewards_array = []
    # full_pre_scaled_starc_rewards_array = []

    # partial_raw_rewards_array = []
    # partial_starc_rewards_array = []
    # partial_pre_scaled_starc_rewards_array = []

    # for i in range(3):

    #     model.freeze()

    #     # Full
    #     full_raw_rewards = model(full_inputs).reshape((2500))
    #     full_starc_rewards, full_pre_scaled_starc_rewards = starc_standardizer.standardize(full_inputs, full_raw_rewards, model, policy)

    #     full_raw_rewards_array.append(full_raw_rewards)
    #     full_starc_rewards_array.append(full_starc_rewards)
    #     full_pre_scaled_starc_rewards_array.append(full_pre_scaled_starc_rewards)

    #     # Partial
    #     for i in range(len(partial_inputs)):
    #         partial_raw_rewards_array.append([])
    #         partial_starc_rewards_array.append([])
    #         partial_pre_scaled_starc_rewards_array.append([])

    #         partial_raw_rewards = model(partial_inputs[i]).reshape((100))
    #         partial_starc_rewards, partial_pre_scaled_starc_rewards = starc_standardizer.standardize(partial_inputs[i], partial_raw_rewards, model, policy)

    #         partial_raw_rewards_array[i].append(partial_raw_rewards)
    #         partial_starc_rewards_array[i].append(partial_starc_rewards)
    #         partial_pre_scaled_starc_rewards_array[i].append(partial_pre_scaled_starc_rewards)
        
    #     model.unfreeze()
    
    # os.makedirs(f"reward_plots", exist_ok=True)
    # plot_full_reward_suit(full_prompt_action_pairs, [full_raw_rewards_array, full_pre_scaled_starc_rewards_array, full_starc_rewards_array],
    #     "reward_plots/full_reward_suit")
    # plot_partial_reward_suit(partial_prompt_action_pairs, [partial_raw_rewards_array, partial_pre_scaled_starc_rewards_array, partial_starc_rewards_array],
    #     "reward_plots/partial_reward_suit")
    

    # plot_reward_scatter_with_seaborn_and_save(prompt_action_pairs, raw_rewards_array,
    #     "reward_plots/raw_rewards")
    # plot_reward_scatter_with_seaborn_and_save(prompt_action_pairs, starc_rewards_array,
    #     "reward_plots/starc_rewards")






