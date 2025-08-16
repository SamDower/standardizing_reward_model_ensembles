from data_selection_policies import DataSelectionPolicy
from scipy.stats import entropy

import torch

class SumVarRSelectionPolicy(DataSelectionPolicy):
    def select_data(self, dataset, preference_model, batch_size):
        """
        Given a dataset with N prompts, generate K answers pairs, rank all N*K points based on uncertainty, and select top B points.
        Prompts are not removed from the pool set since new triples with the same prompt are valid options.

        Args:
            dataset (list): The dataset to select data from.
            N (int): Number of points to select.

        Returns:
            list: List of selected data points.
        """
 

        (trajectories1, trajectories2) = dataset.samples
        stand_reward_variances_table = preference_model.stand_reward_variances_table

        var_sum1 = torch.zeros(trajectories1.shape[0]).to(trajectories1.device)
        var_sum2 = torch.zeros(trajectories2.shape[0]).to(trajectories2.device)
        for t in range(16):
            states1 = trajectories1[:,t,0]
            actions1 = trajectories1[:,t,1]
            var_sum1 += stand_reward_variances_table[states1, actions1, t]
            states2 = trajectories2[:,t,0]
            actions2 = trajectories2[:,t,1]
            var_sum2 += stand_reward_variances_table[states2, actions2, t]
        
        sum_var_scores = var_sum1 + var_sum2
        topk_values, topk_idx = torch.topk(sum_var_scores, batch_size)
        selected_pairs = (trajectories1[topk_idx], trajectories2[topk_idx])
        #selected_data = selected_triples[0]
        return selected_pairs
    
    def _entropy(self, p):
        probs = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        return -(probs * torch.log(probs) + (1.0-probs) * torch.log(1.0-probs))
