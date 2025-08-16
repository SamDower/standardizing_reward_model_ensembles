from data_selection_policies import DataSelectionPolicy
from markov_desicion_processes import MDP
from policies import Policy
from preference_models import PreferenceModel, UncertaintyAwarePreferenceModel
from reward_standardizers import *
#from trainers import Trainer
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import gc, torch
import numpy as np

class PreferenceOptimizationSimulator:
    def __init__(self,
                 markov_decision_process: MDP, 
                 pool_dataset,
                 eval_dataset,
                 data_selection_policy: DataSelectionPolicy,
                 policy: Policy,
                 reward_standardizer : RewardStandardizer,
                 uncertainty_aware_preference_model: UncertaintyAwarePreferenceModel,
                 ground_truth_preference_model: PreferenceModel,
                 trainer,#: Trainer,
                 exp_name,
                 seed):
        """
        Initializes the PreferenceOptimizationSimulator.

        Args:
            dataset (tuple): A tuple containing the train and test datasets.
            data_selection_policy (DataSelectionPolicy): The data selection policy.
            policy (Policy): The policy for optimization.
            uncertainty_aware_preference_model (UncertaintyAwarePreferenceModel): The uncertainty-aware preference model.
            ground_truth_preference_model (GroundTruthPreferenceModel): The ground truth preference model.
        """
        self.markov_decision_process = markov_decision_process
        self.pool_dataset = pool_dataset
        self.test = eval_dataset
        self.data_selection_policy = data_selection_policy
        self.policy = policy
        self.trainer = trainer
        self.reward_standardizer = reward_standardizer
        self.uncertainty_aware_preference_model = uncertainty_aware_preference_model
        self.gt_preference_model = ground_truth_preference_model
        self.exp_name = exp_name
        self.seed = seed

    def run(self, args, debug=True):
        """
        Runs the preference optimization simulation.

        Returns:
            None
        """

        evals_policies = []
        evals_rewards = []
        for epoch in range(args['epochs']):
            print(f"############################ EPOCH {epoch} ################################")
            # 1. Select data using the data selection policy. Pool is updated according to the data selection policy
            trajectory_pairs = self.data_selection_policy.select_data(self.pool_dataset, self.uncertainty_aware_preference_model, batch_size=args['batch_size'])
            # 2. Generate Preferences from Oracle
            preference_labels, _, _, _ = self.markov_decision_process.generate_gt_preferences(trajectory_pairs, add_aleatoric=True)
            preference_tuples = trajectory_pairs + (preference_labels,)
            # 3. Train/Eval the uncertainty-aware preference model
            self.uncertainty_aware_preference_model.train(preference_tuples, full_retrain=args['full_retrain'])

            print("Generating raw table")
            self.uncertainty_aware_preference_model.generate_raw_reward_table()

            # Calculate STARC requirements
            print("Caluculating values for standardizer")
            self.reward_standardizer.calculate_values(self.uncertainty_aware_preference_model)
            print("Calculating norms for standardizer")
            self.reward_standardizer.calculate_norms(self.uncertainty_aware_preference_model)

            print("Generating stand table")
            self.uncertainty_aware_preference_model.generate_stand_reward_table()

            if (epoch+1) % args['evaluate_every'] == 0:


                # Save preference model
                print("Saving reward models")
                for i in range(len(self.uncertainty_aware_preference_model.models)):
                    torch.save(self.uncertainty_aware_preference_model.models[i].state_dict(), 
                            f"experiments/{self.exp_name}/{self.seed}/raw/reward_model_epoch_{epoch}_{i}")
                
                # Evaluate preference model
                print("Evaluating preference model")
                rw_eval_dict = self.uncertainty_aware_preference_model.eval(trajectory_pairs, self.test.samples, self.markov_decision_process)
                evals_rewards.append(rw_eval_dict)

                # 4. Train Policy with learned preference model and Evaluate it
                eval_dict = {}#self.trainer.train_and_evaluate(self.uncertainty_aware_preference_model, self.markov_decision_process, kl_budget=args['kl_budget'])
                evals_policies.append(eval_dict)

                # Plot evaluations
                print("Plotting evaluations")
                plot_reward_evaluation_curves(evals_rewards, self.exp_name, args, self.seed)
                save_latest_full_data(evals_rewards, self.exp_name, args, self.seed)
                #plot_latest_data_stats(evals_rewards, self.exp_name, args, self.seed)
                #plot_latest_reward_tables(evals_rewards, self.exp_name, args, self.seed)

            del trajectory_pairs, preference_labels, preference_tuples
            gc.collect()
            torch.cuda.empty_cache()
            utils.print_gpu_usage("Loop End")

        return evals_policies, evals_rewards


def save_latest_full_data(eval_rewards, exp_name, config, seed):
    latest_eval_dict = eval_rewards[-1]
    epoch = -1 + len(eval_rewards) * config['evaluate_every']
    trajectories1, trajectories2 = latest_eval_dict['full_data']
    utils.tensors_to_csv([trajectories1.flatten(), trajectories2.flatten()], ['Trajectories1', 'Trajectories2'], f"experiments/{exp_name}/{seed}/raw/full_data_{epoch}")
    

def plot_reward_evaluation_curves(eval_rewards, exp_name, config, seed):
    log_likelihoods_train = []
    log_likelihoods_test = []
    accuracies_train = []
    accuracies_test_stand = []
    accuracies_test_raw = []
    #starc_distances = []
    for eval_dict in eval_rewards:
        log_likelihoods_train.append(eval_dict['ll_train'])
        log_likelihoods_test.append(eval_dict['ll_test'])
        accuracies_train.append(eval_dict['acc_train'])
        accuracies_test_stand.append(eval_dict['acc_test_stand'])
        accuracies_test_raw.append(eval_dict['acc_test_raw'])
        #starc_distances.append(eval_dict['starc_distance'])
    
    acquired_data = np.arange(1, len(eval_rewards) + 1) * config['batch_size'] * config['evaluate_every']
    utils.tensors_to_csv([acquired_data, log_likelihoods_train, log_likelihoods_test], ['Acquired Data', 'Log Likelihood (Train)', 'Log Likelihood (Test)'], f"experiments/{exp_name}/{seed}/raw/loglikelihoods")
    utils.tensors_to_csv([acquired_data, accuracies_train, accuracies_test_stand, accuracies_test_raw], ['Acquired Data', 'Accuracy (Train)', 'Accuracy (Test Stand)', 'Accuracy (Test Raw)'], f"experiments/{exp_name}/{seed}/raw/accuracies")
    #utils.tensors_to_csv([acquired_data, starc_distances], ['Acquired Data', 'STARC Distance'], f"experiments/{exp_name}/{seed}/raw/starc_distances")
    
    utils.plot_rate_over_data(acquired_data, log_likelihoods_train, "LogLikelihood", f"experiments/{exp_name}/{seed}/plot/loglikelihood_train")
    utils.plot_rate_over_data(acquired_data, log_likelihoods_test, "LogLikelihood", f"experiments/{exp_name}/{seed}/plot/loglikelihood_test")
    utils.plot_rate_over_data(acquired_data, accuracies_train, "Accuracy", f"experiments/{exp_name}/{seed}/plot/accuracies_train")
    utils.plot_rate_over_data(acquired_data, accuracies_test_stand, "Accuracy", f"experiments/{exp_name}/{seed}/plot/accuracies_test_stand")
    utils.plot_rate_over_data(acquired_data, accuracies_test_raw, "Accuracy", f"experiments/{exp_name}/{seed}/plot/accuracies_test_raw")
    #utils.plot_rate_over_data(acquired_data, starc_distances, "STARC Distance", f"experiments/{exp_name}/{seed}/plot/starc_distances")

def plot_latest_reward_tables(eval_rewards, exp_name, config, seed):

    latest_eval_dict = eval_rewards[-1]
    raw_reward_table = latest_eval_dict['raw_reward_table']
    raw_reward_variances_table = latest_eval_dict['raw_reward_variances_table']
    standardized_reward_table = latest_eval_dict['standardized_reward_table']
    standardized_reward_variances_table = latest_eval_dict['standardized_reward_variances_table']
    epoch = -1 + len(eval_rewards) * config['evaluate_every']
    utils.plot_reward_tables(raw_reward_table, raw_reward_variances_table, standardized_reward_table, standardized_reward_variances_table, epoch, f"experiments/{exp_name}/{seed}/plot/reward_stats/reward_tables")


def plot_latest_data_stats(eval_rewards, exp_name, config, seed):
    latest_eval_dict = eval_rewards[-1]
    data_stats = latest_eval_dict['data_stats']

    hist_states = data_stats['states'][0]
    bins_states = data_stats['states'][1]
    hist_actions = data_stats['actions'][0]
    bins_actions = data_stats['actions'][1]

    acquired_data = np.arange(1, len(eval_rewards) + 1) * config['batch_size'] * config['evaluate_every']
    epoch = -1 + len(eval_rewards) * config['evaluate_every']
    #utils.tensors_to_csv([acquired_data, hist_states, bins_states, hist_actions, bins_actions], ['Acquired Data', 'HistogramStates', 'BinsStates', 'HistogramActions', 'BinsActions'], f"experiments/{exp_name}/{seed}/raw/data_stats")
    plot_dataset_histograms(hist_states, bins_states, hist_actions, bins_actions, f"experiments/{exp_name}/{seed}/plot/data_stats/histogram", epoch)

def plot_dataset_histograms(hist_prompt, bins_prompt, hist_acts, bins_acts, filepath_prefix, epoch):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    values_prompt = []
    for count, bin_start, bin_end in zip(hist_prompt, bins_prompt[:-1], bins_prompt[1:]):
        values_prompt.extend([bin_start] * count)

    values_acts = []
    for count, bin_start, bin_end in zip(hist_acts, bins_acts[:-1], bins_acts[1:]):
        values_acts.extend([bin_start] * count)

    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot the first histogram
    sns.histplot(values_prompt, bins=bins_prompt, kde=False, ax=axes[0])
    axes[0].set_title('States Distribution')
    axes[0].set_xlabel('States Value')
    axes[0].set_ylabel('Frequency')

    sns.heatmap(hist_prompt.reshape(6,6), ax=axes[1], annot=True, annot_kws={'size': 10}, fmt='g')
    axes[1].set_title('States Distribution')

    # Plot the second histogram
    sns.histplot(values_acts, bins=bins_acts, kde=False, ax=axes[2])
    axes[2].set_title('Actions Distribution')
    axes[2].set_xlabel('Action Value')
    axes[2].set_ylabel('Frequency')

    plt.savefig(f"{filepath_prefix}_{epoch}", dpi=300)  # Adjust dpi as needed
    plt.close()