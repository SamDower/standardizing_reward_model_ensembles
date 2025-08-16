from reward_standardizers import RewardStandardizer
from policies import *

import torch
import torch.nn.functional as F

class StarcRewardStandardizer(RewardStandardizer):
        
    def __init__(self, config, device, markov_decision_process) -> None:
        self.args = config
        self.device = device
        self.markov_decision_process = markov_decision_process
        self.value_iterator = ValueIterationFinite(self.markov_decision_process, 16, self.device)
        self.values = [] # [E,T+1,S]
        self.norms = []
        self.mean_raw_mag = 1
        
    def standardize(self, state_action_inputs, t_index, rewards, reward_model_index):
        """
        Computes STARC rewards for prompt_action pairs 

        Args:
            state_action_inputs (torch.tensor): Has shape [n, 2]
            rewards (torch.tensor): Has shape [n, 1]
        """
        if len(self.norms) == 0:
            return rewards

        canonicalised_rewards = self.canonicalise(state_action_inputs, t_index, rewards, reward_model_index)
        normalised_rewards = canonicalised_rewards / self.norms[reward_model_index]
        if self.args['dynamic_magnitude']:
            normalised_rewards = normalised_rewards * self.mean_raw_mag
        if self.args['set_reward_magnitude']:
            normalised_rewards = normalised_rewards * self.args['mag']
        return normalised_rewards
        
    
    def canonicalise(self, state_action_inputs, t_index, rewards, reward_model_index):
        with torch.no_grad():
            # For each prompt
            val_adjust_numbers = torch.tensor([]).to(self.device)
            for i in range(len(state_action_inputs)):
                next_state = self.markov_decision_process.get_next_state(state_action_inputs[i])
                val_current = self.values[reward_model_index][16-t_index][state_action_inputs[i][0]]
                val_next = self.values[reward_model_index][16-t_index-1][next_state]
                val_adjust_num = torch.tensor([-val_current + 1 * val_next]).to(self.device)
                val_adjust_numbers = torch.cat((val_adjust_numbers, val_adjust_num))

            val_adjust_numbers = val_adjust_numbers.reshape(rewards.shape)
            canonicalised_rewards = rewards + val_adjust_numbers

            del val_adjust_numbers
            return canonicalised_rewards

    def calculate_values(self, uncertainty_aware_preference_model):
        values_array = []
        for i in range(len(uncertainty_aware_preference_model.models)):
            values_array.append(self.value_iterator.perform_value_iteration_on_model(uncertainty_aware_preference_model.models[i]))
        self.values = torch.stack(values_array, dim=0)

    def calculate_norms(self, uncertainty_aware_preference_model):
        
        self.norms = []
        repeated_states = torch.linspace(0, 35, 36).to(torch.int64).to(self.device).repeat(4)
        repeated_actions = torch.linspace(0, 3, 4).to(torch.int64).to(self.device).repeat_interleave(36)
        all_state_actions = torch.cat((repeated_states.reshape(36*4, 1), repeated_actions.reshape(36*4, 1)), dim=1)
        print(f"All state actions shape {all_state_actions.shape}")

        for i in range(len(uncertainty_aware_preference_model.models)):
            # trajectory_rewards = uncertainty_aware_preference_model.compute_trajectory_rewards_from_model( 
            #     eval_data[0], i, use_standardizer=False)
            # rewards = uncertainty_aware_preference_model.compute_state_action_rewards_from_model(all_state_actions, i, use_standardizer=False)
            # canonicalised_rewards = self.canonicalise(all_state_actions, rewards, i)
            canonicalised_rewards_array = []
            for t in range(16):
                rewards = uncertainty_aware_preference_model.compute_state_action_rewards_from_model(all_state_actions, t, i, use_standardizer=False)
                canonicalised_rewards_array.append(self.canonicalise(all_state_actions, t, rewards, i))
            #canonicalised_rewards = torch.stack([self.canonicalise(all_state_actions, rewards, i) for t in range(16+1)], dim = 0)
            canonicalised_rewards = torch.stack(canonicalised_rewards_array, dim = 0)
            self.norms.append(torch.sum(torch.abs(canonicalised_rewards)))
        
        if self.args['dynamic_magnitude']:
            extended_table = uncertainty_aware_preference_model.raw_reward_table.mean(dim=0).reshape(36,4,1).expand(-1,-1,16)
            print(extended_table.shape)
            self.mean_raw_mag = torch.sum(torch.abs(extended_table))
            print(self.mean_raw_mag)

