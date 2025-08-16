from data_selection_policies import DataSelectionPolicy
from scipy.stats import entropy

import torch

class VarRSelectionPolicy(DataSelectionPolicy):
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
 
        # Randomly select N points
        # data = dataset.samples
        # repeated_dataset = data.repeat(self.args['action_pairs_number'])
        # action_one = policy.act(repeated_dataset)
        # action_two = policy.act(repeated_dataset)
        # triples = (repeated_dataset, action_one, action_two)

        data = dataset.samples
        _, pref_probs, rewards_first_samples, rewards_second_samples, _ = preference_model.generate_preferences(data, use_standardizer=True, use_table=True)
        
        #bald_scores = self._entropy(torch.mean(pref_probs_samples, axis=1)) - torch.mean(self._entropy(pref_probs_samples))
        var_scores = (rewards_first_samples - rewards_second_samples).var(axis=1)
        topk_values, topk_idx = torch.topk(var_scores, batch_size)
        selected_pairs = (data[0][topk_idx], data[1][topk_idx])
        #selected_data = selected_triples[0]
        return selected_pairs
    
    def _entropy(self, p):
        probs = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        return -(probs * torch.log(probs) + (1.0-probs) * torch.log(1.0-probs))
