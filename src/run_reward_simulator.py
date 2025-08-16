from trajectory_datasets import *
from markov_desicion_processes import *
from data_selection_policies import *
from trainers import *
from reward_models import *
from policies import *
from reward_standardizers import *
from simulators import RewardOptimizationSimulator
from configs import load_config, override_config
import utils
import argparse

import torch
import numpy as np
import os


def from_class(classname):
    import sys
    return getattr(sys.modules[__name__], classname)

def main():
    print("reached main")
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--config_file', type=str, default='iterative_exponential_density')
    args = parser.parse_args()


    base_config = load_config("base_config")

    if args.config_file:
        config = load_config(args.config_file)
        print(f"Overriding base config with {args.config_file}")
        config = override_config(base_config, config)
    seed = config['seed'] if args.seed is None else args.seed
    
    utils.set_seed(seed)
    device = torch.device('cuda')

    markov_decision_process = from_class(config['mdp']['class'])(config['mdp'], device)

    print("creating train set")
    dataset = from_class(config['train_dataset']['class'])(config['train_dataset'], markov_decision_process, device=device)
    dataset.generate_dataset(num_samples=config['data']['train_size'])
    
    print("creating test set")
    eval_dataset = from_class(config['eval_dataset']['class'])(config['eval_dataset'], markov_decision_process, device=device)
    eval_dataset.generate_dataset(num_samples=config['data']['test_size'])
    
    print("setting up rest")
    data_selection_policy = from_class(config['data_selection']['policy'])(config['data_selection'])
    policy = None # from_class(config['policy']['class'])(config['policy'], device=device)
    policy_trainer = from_class(config['trainer']['class'])(config['trainer'], device=device)

    reward_standardizer = from_class(config['reward_standardizer']['class'])(config['reward_standardizer'], device, markov_decision_process)

    uncertainty_aware_reward_model = from_class(config['pref_model']['class'])(device, reward_standardizer, **config['pref_model'])

    print("---------Config----------")
    print(f"epochs: {config['simulation']['epochs']}")
    print(f"exp_name: {config['exp_name']}")
    print(f"train_dataset: {config['train_dataset']['class']}")
    print(f"train_dataset: {config['eval_dataset']['class']}")
    print(f"pref_model: {config['pref_model']['class']}")
    print(f"standardizer: {config['reward_standardizer']['class']}")
    print("-------------------------")

    # Create the simulator
    simulator = RewardOptimizationSimulator(
        markov_decision_process,
        dataset.samples[0],
        eval_dataset,
        data_selection_policy,
        policy,
        reward_standardizer,
        uncertainty_aware_reward_model,
        policy_trainer,
        config['exp_name'],
        seed
    )

    # Run the simulation
    exp_name = config['exp_name']
    os.makedirs(f"experiments/{exp_name}/{seed}", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/{seed}/plot", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/{seed}/plot/data_stats", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/{seed}/plot/reward_stats", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/{seed}/raw", exist_ok=True)
    eval_policies, eval_rewards = simulator.run(config['simulation'])
    
    
    #plot_policy_evaluation(eval_policies, exp_name, config, seed)
    # plot_reward_evaluation(eval_rewards, exp_name, config, seed)
    # plot_data_stats(eval_rewards, exp_name, config, seed)

def plot_policy_evaluation(eval_policies, exp_name, config, seed):
    epoch = 0
    win_rates = []
    optimal_rewards = []
    # win_rates_train = []
    # win_rates_test = []
    # cvar_wrs_train = []
    # cvar_wrs_test = []
    for eval_dict in eval_policies:
        # r_proxy_train, r_gt_train = eval_dict['train']
        # r_proxy_test, r_gt_test = eval_dict['test']
        # kl_budget = eval_dict['kl']
        win_rate = eval_dict['win_rate']
        optimal_reward = eval_dict['optimal_reward']
        # win_rate_train = eval_dict['win_rate_train']
        # win_rate_test = eval_dict['win_rate_test']
        # cvar_wr_train = eval_dict['cvar_win_rate_train']
        # cvar_wr_test = eval_dict['cvar_win_rate_test']

        win_rates.append(win_rate)
        optimal_rewards.append(optimal_reward)
        # win_rates_train.append(win_rate_train)
        # win_rates_test.append(win_rate_test)
        # cvar_wrs_train.append(cvar_wr_train)
        # cvar_wrs_test.append(cvar_wr_test)

        # Output csv
        # utils.tensors_to_csv([kl_budget, r_proxy_train, r_gt_train], ['KL Budget', 'Proxy', 'GroundTruth'], filename=f"experiments/{exp_name}/{seed}/raw/toy_overoptimization_train_epoch_{epoch}")
        # utils.tensors_to_csv([kl_budget, r_proxy_test, r_gt_test], ['KL Budget', 'Proxy', 'GroundTruth'], filename=f"experiments/{exp_name}/{seed}/raw/toy_overoptimization_eval_epoch_{epoch}")

        # Plot from csv
        # utils.plot_curves_with_seaborn_and_save(csv_file=f"experiments/{exp_name}/{seed}/raw/toy_overoptimization_train_epoch_{epoch}",
        #                                         plot_file=f"experiments/{exp_name}/{seed}/plot/toy_overoptimization_train_epoch_{epoch}")
        # utils.plot_curves_with_seaborn_and_save(csv_file=f"experiments/{exp_name}/{seed}/raw/toy_overoptimization_eval_epoch_{epoch}",
        #                                         plot_file=f"experiments/{exp_name}/{seed}/plot/toy_overoptimization_eval_epoch_{epoch}")
        # epoch += 1
    
    x = np.arange(config['simulation']['epochs']) * config['simulation']['batch_size']
    utils.to_csv(["Acquired Data", "Win Rate"], [x, win_rates], f"experiments/{exp_name}/{seed}/raw/win_rates")
    utils.plot_rate_over_data(x, win_rates, "Win Rate", f"experiments/{exp_name}/{seed}/plot/win_rate")
    # utils.to_csv(["Acquired Data", "Win Rate (Train)", "Win Rate (Test)"], [x, win_rates_train, win_rates_test], f"experiments/{exp_name}/{seed}/raw/win_rates")
    # utils.plot_rate_over_data(x, win_rates_train, "Win Rate", f"experiments/{exp_name}/{seed}/plot/win_rate_train")
    # utils.plot_rate_over_data(x, win_rates_test, "Win Rate", f"experiments/{exp_name}/{seed}/plot/win_rate_test")

    utils.to_csv(["Acquired Data", "Optimal Reward"], [x, optimal_rewards], f"experiments/{exp_name}/{seed}/raw/optimal_rewards")
    utils.plot_rate_over_data(x, optimal_rewards, "Optimal Reward", f"experiments/{exp_name}/{seed}/plot/optimal_rewards")

    # #CVaR
    # utils.to_csv(["Acquired Data", "Win Rate (CVaR - Train)", "Win Rate (CVaR - Test)"], [x, cvar_wrs_train, cvar_wrs_test], f"experiments/{exp_name}/{seed}/raw/cvar_win_rates")
    # utils.plot_rate_over_data(x, cvar_wrs_train, "Win Rate (CVaR)", f"experiments/{exp_name}/{seed}/plot/cvar_win_rate_train")
    # utils.plot_rate_over_data(x, cvar_wrs_test, "Win Rate (CVaR)", f"experiments/{exp_name}/{seed}/plot/cvar_win_rate_test")

def plot_reward_evaluation(eval_rewards, exp_name, config, seed):
    epoch = 0
    #log_likelihoods_train = []
    #log_likelihoods_test = []
    accuracies_train = []
    accuracies_test_stand = []
    accuracies_test_raw = []
    for eval_dict in eval_rewards:
        # rw_train = eval_dict['rewards_train']
        # rw_test = eval_dict['rewards_test']
        # rw_test_raw = eval_dict['rewards_test_raw']

        #ll_train = eval_dict['ll_train']
        #ll_test = eval_dict['ll_test']
        acc_train = eval_dict['acc_train']
        acc_test_stand = eval_dict['acc_test_stand']
        acc_test_raw = eval_dict['acc_test_raw']

        #log_likelihoods_train.append(ll_train)
        #log_likelihoods_test.append(ll_test)
        accuracies_train.append(acc_train)
        accuracies_test_stand.append(acc_test_stand)
        accuracies_test_raw.append(acc_test_raw)

        # # Output csv
        # utils.tensors_to_csv(rw_train, ['Prompt', 'Action', 'RewardsProxy', 'RewardsGroundTruth'], filename=f"experiments/{exp_name}/{seed}/raw/rewards_train_epoch_{epoch}")
        # utils.tensors_to_csv(rw_test, ['Prompt', 'Action', 'RewardsProxy', 'RewardsGroundTruth'], filename=f"experiments/{exp_name}/{seed}/raw/rewards_test_epoch_{epoch}")
        # utils.tensors_to_csv(rw_test_raw, ['Prompt', 'Action', 'RewardsProxy', 'RewardsGroundTruth'], filename=f"experiments/{exp_name}/{seed}/raw/rewards_test_raw_epoch_{epoch}")

        # # Plot from csv
        # utils.plot_reward_scatter_with_seaborn_and_save(csv_file=f"experiments/{exp_name}/{seed}/raw/rewards_train_epoch_{epoch}",
        #                                                plot_file=f"experiments/{exp_name}/{seed}/plot/rewards_scatter_train_epoch_{epoch}")
        # utils.plot_reward_scatter_with_seaborn_and_save(csv_file=f"experiments/{exp_name}/{seed}/raw/rewards_test_epoch_{epoch}",
        #                                                plot_file=f"experiments/{exp_name}/{seed}/plot/rewards_scatter_test_epoch_{epoch}")
        # utils.plot_reward_scatter_with_seaborn_and_save(csv_file=f"experiments/{exp_name}/{seed}/raw/rewards_test_raw_epoch_{epoch}",
        #                                                 plot_file=f"experiments/{exp_name}/{seed}/plot/rewards_scatter_test_epoch_{epoch}_raw")
        # utils.plot_reward_curve_with_seaborn_and_save(csv_file=f"experiments/{exp_name}/{seed}/raw/rewards_test_epoch_{epoch}",
        #                                                plot_file=f"experiments/{exp_name}/{seed}/plot/rewards_curve_test_epoch_{epoch}")
        # utils.plot_reward_curve_with_seaborn_and_save(csv_file=f"experiments/{exp_name}/{seed}/raw/rewards_test_raw_epoch_{epoch}",
        #                                                plot_file=f"experiments/{exp_name}/{seed}/plot/rewards_curve_test_epoch_{epoch}_raw")
        
        raw_reward_table = eval_dict['raw_reward_table']
        raw_reward_variances_table = eval_dict['raw_reward_variances_table']
        standardized_reward_table = eval_dict['standardized_reward_table']
        standardized_reward_variances_table = eval_dict['standardized_reward_variances_table']
        utils.plot_reward_tables(raw_reward_table, raw_reward_variances_table, standardized_reward_table, standardized_reward_variances_table, epoch, f"experiments/{exp_name}/{seed}/plot/reward_stats/reward_tables")

        epoch += 1
    
    acquired_data = np.arange(1, config['simulation']['epochs'] + 1) * config['simulation']['batch_size']
    #utils.tensors_to_csv([acquired_data, log_likelihoods_train, log_likelihoods_test], ['Acquired Data', 'Log Likelihood (Train)', 'Log Likelihood (Test)'], f"experiments/{exp_name}/{seed}/raw/loglikelihoods")
    utils.tensors_to_csv([acquired_data, accuracies_train, accuracies_test_stand, accuracies_test_raw], ['Acquired Data', 'Accuracy (Train)', 'Accuracy (Test Stand)', 'Accuracy (Test Raw)'], f"experiments/{exp_name}/{seed}/raw/accuracies")
    
    #utils.plot_rate_over_data(acquired_data, log_likelihoods_train, "LogLikelihood", f"experiments/{exp_name}/{seed}/plot/loglikelihood_train")
    #utils.plot_rate_over_data(acquired_data, log_likelihoods_test, "LogLikelihood", f"experiments/{exp_name}/{seed}/plot/loglikelihood_test")
    utils.plot_rate_over_data(acquired_data, accuracies_train, "Accuracy", f"experiments/{exp_name}/{seed}/plot/accuracies_train")
    utils.plot_rate_over_data(acquired_data, accuracies_test_stand, "Accuracy", f"experiments/{exp_name}/{seed}/plot/accuracies_test_stand")
    utils.plot_rate_over_data(acquired_data, accuracies_test_raw, "Accuracy", f"experiments/{exp_name}/{seed}/plot/accuracies_test_raw")


def plot_data_stats(eval_rewards, exp_name, config, seed):
    hist_states, bins_states = [], []
    hist_actions, bins_actions = [], []
    for eval_dict in eval_rewards:
        data_stats = eval_dict['data_stats']

        hist_states.append(data_stats['states'][0])
        bins_states.append(data_stats['states'][1])
        hist_actions.append(data_stats['actions'][0])
        bins_actions.append(data_stats['actions'][1])
    
    acquired_data = np.arange(1, config['simulation']['epochs'] + 1) * config['simulation']['batch_size']
    utils.tensors_to_csv([acquired_data, hist_states, bins_states, hist_actions, bins_actions], ['Acquired Data', 'HistogramStates', 'BinsStates', 'HistogramActions', 'BinsActions'], f"experiments/{exp_name}/{seed}/raw/data_stats")
    utils.plot_dataset_histograms(acquired_data, hist_states, bins_states, hist_actions, bins_actions, f"experiments/{exp_name}/{seed}/plot/data_stats/histogram")


if __name__ == "__main__":
    main()
