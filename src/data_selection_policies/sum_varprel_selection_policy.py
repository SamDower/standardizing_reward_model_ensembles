from data_selection_policies import DataSelectionPolicy
from scipy.stats import entropy

import torch
import numpy as np

class SumVarPSelectionPolicy(DataSelectionPolicy):
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
        diff_vars = var_sum1 + var_sum2

        rewards1, _= preference_model.generate_reward(trajectories1, use_standardizer=True, use_table=True)
        rewards2, _ = preference_model.generate_reward(trajectories2, use_standardizer=True, use_table=True)
        diff_means = rewards1 - rewards2

        ex_sigmoid_X = torch.sigmoid(diff_means / torch.sqrt(1 + (torch.pi*torch.pi * diff_vars)/8))

        ex_sigmoid_squared_X = []
        for i in range(trajectories1.shape[0]):
            # Sample from N(mu, sigma^2)
            samples = np.random.normal(diff_means[i].detach().cpu(), np.sqrt(diff_vars[i].detach().cpu()), size=1000)
            sigmoid_sq_samples = (1 / (1 + np.exp(-samples)))**2

            # Estimate expectation
            estimate = np.mean(sigmoid_sq_samples)
            ex_sigmoid_squared_X.append(estimate)
        ex_sigmoid_squared_X = torch.tensor(ex_sigmoid_squared_X).to(trajectories1.device)

        var_sigmoid_X_scores = ex_sigmoid_squared_X - torch.square(ex_sigmoid_X)

        topk_values, topk_idx = torch.topk(var_sigmoid_X_scores, batch_size)
        selected_pairs = (trajectories1[topk_idx], trajectories2[topk_idx])
        #selected_data = selected_triples[0]
        return selected_pairs
    
    def _entropy(self, p):
        probs = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        return -(probs * torch.log(probs) + (1.0-probs) * torch.log(1.0-probs))
