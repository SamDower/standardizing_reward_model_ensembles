from policies import Policy
import torch
import numpy as np
import torch.nn.functional as F

class ValueIterationFinite:
    def __init__(self, markov_decision_process, horizon_T, device):
        """
        """
        self.markov_decision_process = markov_decision_process
        self.device = device
        self.horizon_T = horizon_T
        self.values = torch.zeros(horizon_T+1, 36).to(device)

    def perform_value_iteration_on_mdp(self):
        self.values = torch.zeros(self.horizon_T+1, 36).to(self.device)
        reward_table = torch.zeros(36, 4).to(self.device)
        for s in range(36):
            for a in range(4):
                reward_table[s][a] = self.markov_decision_process.get_reward(torch.tensor([s, a]).to(self.device))
        return self.perform_value_iteration(reward_table)

    def perform_value_iteration_on_model(self, model):
        self.values = torch.zeros(self.horizon_T+1, 36).to(self.device)
        reward_table = torch.zeros(36, 4).to(self.device)
        for s in range(36):
            state_actions = torch.tensor([[s,0],[s,1],[s,2],[s,3]]).to(self.device)
            one_hot_inputs = torch.cat((F.one_hot(state_actions[:,0], num_classes=36), F.one_hot(state_actions[:,1], num_classes=4)), dim = 1).float()
            rewards = model(one_hot_inputs)
            for a in range(4):
                reward_table[s][a] = rewards[a]
        return self.perform_value_iteration(reward_table)

    def perform_value_iteration(self, reward_table):
        # Assumes uniform random policy
        for h in range(1, self.horizon_T+1):
            for s in range(36):
                new_value = 0
                for a in range(4):
                    sa = torch.tensor([s, a]).to(self.device)
                    new_value += 0.25 * (reward_table[sa[0]][sa[1]] + 1 * self.values[h-1][self.markov_decision_process.get_next_state(sa)])
                self.values[h][s] = new_value
        return self.values