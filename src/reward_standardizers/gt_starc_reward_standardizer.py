from reward_standardizers import RewardStandardizer
from policies import *

import torch
import torch.nn.functional as F

class GTStarcRewardStandardizer(RewardStandardizer):
        
    def __init__(self, device, markov_decision_process) -> None:
        self.device = device
        self.markov_decision_process = markov_decision_process
        self.values = []
        self.norm = 0
        
    def standardize(self, state_action_inputs, t_index):
        """
        Computes STARC rewards for prompt_action pairs 

        Args:
            state_action_inputs (torch.tensor): Has shape [n, 2]
            rewards (torch.tensor): Has shape [n, 1]
        """

        with torch.no_grad():
            canonicalised_rewards = self.canonicalise(state_action_inputs, t_index)
            normalised_rewards = canonicalised_rewards / self.norm

            del canonicalised_rewards
            return normalised_rewards
    
    def canonicalise(self, state_action_inputs, t_index):
        # For each prompt
        cannonicalised_rewards = torch.tensor([]).to(self.device)
        for i in range(len(state_action_inputs)):

            next_state = self.markov_decision_process.get_next_state(state_action_inputs[i])
            val_adjust_num = torch.tensor([-self.values[16-t_index][state_action_inputs[i][0]] + self.values[16-t_index-1][next_state]]).to(self.device)
            cann = self.markov_decision_process.get_reward(state_action_inputs[i]) + val_adjust_num
            cannonicalised_rewards = torch.cat((cannonicalised_rewards, cann))

        return cannonicalised_rewards.reshape(state_action_inputs.shape[0], 1)
    
    def calculate_values(self):
        iterator = ValueIterationFinite(self.markov_decision_process, 16, self.device)
        self.values = iterator.perform_value_iteration_on_mdp()

    def calculate_norms(self):
        rewards = torch.zeros(36,4,16)
        for s in range(36):
            for a in range(4):
                sa = torch.tensor([s,a]).reshape(1,2).to(self.device)
                for t in range(16):
                    rewards[s][a][t] = self.canonicalise(sa, t)
        self.norm = torch.sum(torch.abs(rewards))
        

