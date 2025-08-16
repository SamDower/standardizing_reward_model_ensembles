from data_selection_policies import DataSelectionPolicy
from scipy.stats import entropy

import torch

class SingleSumVarRSelectionPolicy(DataSelectionPolicy):
    def select_data(self, trajectories, preference_model, batch_size):
        """
        Given a dataset with N prompts, generate K answers pairs, rank all N*K points based on uncertainty, and select top B points.
        Prompts are not removed from the pool set since new triples with the same prompt are valid options.

        Args:
            dataset (list): The dataset to select data from.
            N (int): Number of points to select.

        Returns:
            list: List of selected data points.
        """
 

        stand_reward_variances_table = preference_model.stand_reward_variances_table

        var_sum = torch.zeros(trajectories.shape[0]).to(trajectories.device)
        for t in range(16):
            states = trajectories[:,t,0]
            actions = trajectories[:,t,1]
            var_sum += stand_reward_variances_table[states, actions, t]
        
        topk_values, topk_idx = torch.topk(var_sum, batch_size)
        selected_pairs = trajectories[topk_idx]
        #selected_data = selected_triples[0]
        return selected_pairs
    
    def _entropy(self, p):
        probs = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        return -(probs * torch.log(probs) + (1.0-probs) * torch.log(1.0-probs))
