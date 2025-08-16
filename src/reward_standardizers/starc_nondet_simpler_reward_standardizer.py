from reward_standardizers import RewardStandardizer
from policies import *

import torch
import torch.nn.functional as F

class StarcNonDetSimplerRewardStandardizer(RewardStandardizer):
        
    def __init__(self, config, device, markov_decision_process) -> None:
        self.args = config
        self.device = device
        self.markov_decision_process = markov_decision_process
        self.value_iterator = ValueIterationFiniteNonDet(self.markov_decision_process, 16, self.device)
        self.values = [] # [E,T+1,S]
        self.norms = []
        self.mean_raw_mag = 1
        
    def standardize(self, state_action_inputs, t_index, rewards, reward_model_index, nondet=False):
        """
        Computes STARC rewards for prompt_action pairs 

        Args:
            state_action_inputs (torch.tensor): Has shape [n, 3]
            rewards is a raw rewards table: Has shape [S, A, S]
        """
        if len(self.norms) == 0:
            return rewards

        canonicalised_rewards = self.canonicalise(state_action_inputs, t_index, rewards, reward_model_index)
        normalised_rewards = canonicalised_rewards / self.norms[reward_model_index]
        if self.args['dynamic_magnitude']:
            normalised_rewards = normalised_rewards * self.mean_raw_mag
        return normalised_rewards
        
    
    def canonicalise(self, state_action_inputs, t_index, rewards, reward_model_index):
        """ rewards is a raw rewards table"""
        with torch.no_grad():

            states = state_action_inputs[:,0]
            actions = state_action_inputs[:,1]
            next_states = state_action_inputs[:,2]

            val_current = self.values[reward_model_index][16-t_index][states]
            val_next = self.values[reward_model_index][16-t_index-1][next_states]
            val_adjusted = (val_next - val_current).reshape(state_action_inputs.shape[0])

            reward_term = r_term = rewards[states, actions, next_states] 
            
            canonicalised_rewards = reward_term + val_adjusted
            return canonicalised_rewards

            possible_next_states = self.markov_decision_process.possible_next_states_matrix[states, actions] # [N,4]

            canonicalised_rewards = torch.zeros(state_action_inputs.shape[0]).to(self.device)
            for i in range(4):
                next_states = possible_next_states[:,i]
                r_term = rewards[states, actions, next_states] 
                first_val_term = self.values[reward_model_index][16-t_index][states]
                second_val_term = self.values[reward_model_index][16-t_index-1][next_states]
                transition_prob = 0.7 if i == 0 else 0.1
                canonicalised_rewards += transition_prob * (r_term - first_val_term + second_val_term)
            
            return canonicalised_rewards

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
            values_array.append(self.value_iterator.perform_value_iteration(uncertainty_aware_preference_model.raw_reward_table[i,:,:,:]))
        self.values = torch.stack(values_array, dim=0)

    def calculate_norms(self, uncertainty_aware_preference_model):
        
        self.norms = []
        # repeated_states = torch.linspace(0, 35, 36).to(torch.int64).to(self.device).repeat(4)
        # repeated_actions = torch.linspace(0, 3, 4).to(torch.int64).to(self.device).repeat_interleave(36)
        # all_state_actions = torch.cat((repeated_states.reshape(36*4, 1), repeated_actions.reshape(36*4, 1)), dim=1)
        # all_state_actions = torch.zeros(36*4*4,3)
        sas = []
        for s in range(36):
            for a in range(4):
                for i in range(4):
                    next_s = self.markov_decision_process.possible_next_states_matrix[s][a][i]
                    sas.append([s,a,next_s])
        all_state_actions = torch.tensor(sas).to(torch.int64).to(self.device)
        print(f"All state actions shape {all_state_actions.shape}")

        for i in range(len(uncertainty_aware_preference_model.models)):
            # trajectory_rewards = uncertainty_aware_preference_model.compute_trajectory_rewards_from_model( 
            #     eval_data[0], i, use_standardizer=False)
            # rewards = uncertainty_aware_preference_model.compute_state_action_rewards_from_model(all_state_actions, i, use_standardizer=False)
            # canonicalised_rewards = self.canonicalise(all_state_actions, rewards, i)
            canonicalised_rewards_array = []
            for t in range(16):
                rewards = uncertainty_aware_preference_model.compute_state_action_rewards_from_model(all_state_actions, t, i, use_standardizer=False)
                canonicalised_rewards_array.append(self.canonicalise(all_state_actions, t, uncertainty_aware_preference_model.raw_reward_table[i], i))
            #canonicalised_rewards = torch.stack([self.canonicalise(all_state_actions, rewards, i) for t in range(16+1)], dim = 0)
            canonicalised_rewards = torch.stack(canonicalised_rewards_array, dim = 0)
            self.norms.append(torch.sum(torch.abs(canonicalised_rewards)))
        
        if self.args['dynamic_magnitude']:
            extended_table = uncertainty_aware_preference_model.raw_reward_table.mean(dim=0).reshape(36,4,36,1).expand(-1,-1,-1,16)
            print(extended_table.shape)
            self.mean_raw_mag = torch.sum(torch.abs(extended_table))
            print(self.mean_raw_mag)

