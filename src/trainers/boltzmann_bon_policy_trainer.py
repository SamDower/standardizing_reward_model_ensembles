from trainers import Trainer
from policies import BoltzmannPolicy
import math
import torch

class BoltzmannBoNPolicyTrainer(Trainer):
    def __init__(self, config, device):
        self.args = config
        self.device = device
        
    def train_and_evaluate(self, preference_model, markov_decision_process, kl_budget=2.0):
        """
        Trains the policy using the given data and reward model.

        """
        policy = BoltzmannPolicy(None, markov_decision_process, self.device)
        policy.train_sarsa_on_preference_model(preference_model)

        max_n, kls = self._compute_n_from_kl_budget(kl_budget, device=self.device)
        print(f"KL: {kl_budget}, N: {max_n}")

        bon_trajectories = self._compute_bon_trajectories(100, policy, preference_model, markov_decision_process, max_n)
        win_rate = self._win_rate_eval(policy, bon_trajectories, markov_decision_process)
        
        optimal_reward = markov_decision_process._generate_gt_trajectory_reward(policy.act_out_trajectories(1)).item()
        
        eval_dict = {'win_rate': win_rate, 'optimal_reward': optimal_reward}

        return eval_dict
    
    def _compute_bon_trajectories(self, batch_size, policy, preference_model, markov_decision_process, max_n):
        bon_trajectories = torch.zeros(batch_size, markov_decision_process.tMax, 2, dtype=torch.int64).to(self.device)
        print(f"Computing BoN trajectories. N = {max_n}, batch_size = {batch_size}")
        for i in range(batch_size):
            if i % 10 == 0:
                print(f"On batch {i}")

            trajectories = policy.act_boltzmann_out_trajectories(max_n)
            rewards, _ = preference_model.generate_reward(trajectories, use_standardizer=False)
            sorted_rewards, sorted_rewards_ids = torch.sort(rewards)

            sorted_trajectories = trajectories[sorted_rewards_ids]
            selected_trajectory = sorted_trajectories[-1]

            bon_trajectories[i] = selected_trajectory

        return bon_trajectories

        
    def _win_rate_eval(self, policy, bon_trajectories, markov_decision_process):
        trajectories = policy.act_boltzmann_out_trajectories(bon_trajectories.shape[0])
        pairs = (bon_trajectories, trajectories)
        prefs, _, _, _ = markov_decision_process.generate_gt_preferences(pairs)
        win_rate = prefs.sum() / prefs.shape[0]
        return win_rate.detach().cpu().numpy()


    # def _compute_cvar_win_rate(self, r_gt_train, actions_train, prompts, base_policy, gt_pref_model, alpha=0.1):
    #     alpha_reward = torch.quantile(r_gt_train, alpha)
    #     mask = r_gt_train <= alpha_reward
    #     selected_actions = actions_train[mask]
    #     selected_prompts = prompts[mask]
    #     base_actions = base_policy.act(selected_prompts)
    #     triples = (selected_prompts, selected_actions, base_actions)
    #     prefs, _, _, _ = gt_pref_model.generate_preferences(triples)
    #     win_rate = prefs.sum() / prefs.shape[0]
    #     return win_rate.detach().cpu().numpy()

    def _compute_n_from_kl_budget(self, kl_budget, device):
        kl_spent = 0.0
        n = 0
        kls = []
        while kl_spent <= kl_budget:
            n = n + 1
            kl_spent = self._compute_kl(n)
            kls.append(kl_spent)

        return n - 1, kls[:-1]
    
    def _compute_kl(self, n: torch.Tensor):
        return torch.log(n) - (n - 1) / n
    
    def _compute_kl(self, n: int):
        return math.log(n) - (n - 1) / n

