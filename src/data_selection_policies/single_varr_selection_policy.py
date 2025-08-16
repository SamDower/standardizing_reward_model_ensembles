from data_selection_policies import DataSelectionPolicy
from scipy.stats import entropy

import torch

class SingleVarRSelectionPolicy(DataSelectionPolicy):
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

        _, all_rewards = preference_model.generate_reward(trajectories, use_standardizer=True, use_table=True)
        
        var_scores = all_rewards.var(axis=1)
        topk_values, topk_idx = torch.topk(var_scores, batch_size)
        selected_pairs = trajectories[topk_idx]
        return selected_pairs
    
    def _entropy(self, p):
        probs = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        return -(probs * torch.log(probs) + (1.0-probs) * torch.log(1.0-probs))
