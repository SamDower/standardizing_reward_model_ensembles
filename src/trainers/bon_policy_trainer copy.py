from trainers import Trainer
import math
import torch

class BoNPolicyTrainer(Trainer):
    def __init__(self, config, device):
        self.args = config
        self.device = device
        
    def train_and_evaluate(self, train_prompts, eval_prompts, base_policy, preference_model, gt_preference_model, full_retrain=True, kl_budget=2.0):
        """
        Trains the policy using the given data and reward model.

        """
        device = train_prompts.device
        max_n, kls = self._compute_n_from_kl_budget(kl_budget, device=device)
        print(f"KL: {kl_budget}, N: {max_n}")
        max_n_pt = torch.tensor([max_n]).to(device)

        r_proxy_train, r_gt_train, actions_train, gt_rewards_train = self._train_and_eval(train_prompts, base_policy, preference_model, gt_preference_model, max_n, max_n_pt, "train", device)
        r_proxy_ood, r_gt_ood, actions_ood, gt_rewards_ood = self._train_and_eval(eval_prompts[0], base_policy, preference_model, gt_preference_model, max_n, max_n_pt, "test", device)
        win_rate_train = self._win_rate_eval(train_prompts, base_policy, actions_train, gt_preference_model)
        win_rate_ood = self._win_rate_eval(eval_prompts[0], base_policy, actions_ood, gt_preference_model)
        cvar_win_rate_train = self._compute_cvar_win_rate(gt_rewards_train, actions_train, train_prompts, base_policy, gt_preference_model, alpha=0.1)
        cvar_win_rate_ood = self._compute_cvar_win_rate(gt_rewards_ood, actions_ood, eval_prompts[0], base_policy, gt_preference_model, alpha=0.1)
        eval_dict = {'train': (r_proxy_train, r_gt_train), 'test': (r_proxy_ood, r_gt_ood), 'kl': torch.tensor(kls).to(device), \
                     'win_rate_train': win_rate_train, 'win_rate_test': win_rate_ood, 'cvar_win_rate_train': cvar_win_rate_train, \
                        'cvar_win_rate_test': cvar_win_rate_ood}

        return eval_dict
        

    def _train_and_eval(self, prompts, policy, preference_model, gt_preference_model, max_n, max_n_pt, mode, device):
        all_prompts_gt_estimates = []
        all_prompts_proxy_estimates = []
        selected_actions = []
        selected_gt_rewards = []
        print(f"Training and evaluating mode: {mode}, batch size: {len(prompts)}")
        for prompt in prompts:
            repeated_prompts = prompt.repeat(max_n) # (max_n, prompt_fts)
            actions = policy.act(repeated_prompts) # (max_n, act_fts)
            reward_preds, reward_samples = preference_model.generate_reward((repeated_prompts, actions), policy_for_starc=policy)
            objective = self._compute_objective(reward_preds, reward_samples)
            sorted_rewards, sorted_rewards_ids = torch.sort(objective)

            sorted_actions = actions[sorted_rewards_ids]
            selected_action = sorted_actions[-1]
            gt_rewards = gt_preference_model.generate_reward((repeated_prompts, sorted_actions))
            selected_gt_reward = gt_rewards[-1]
            recentered_gt_rewards = self._recenter_rewards(gt_rewards)
            

            prompt_gt_estimates = []
            prompt_proxy_estimates = []
            for n in range(1, max_n + 1):
                i = torch.arange(n, max_n + 1).to(device)
                n_pt = torch.tensor([n]).to(device)
                # Nakano Estimator for GT Rewards
                curr_estimate_gt = self._torch_nakano_coeff(i - 1, n_pt - 1, max_n_pt, n_pt) * recentered_gt_rewards[i - 1]
                estimate_gt = curr_estimate_gt.sum()

                # Nakano Estimator for Proxy Rewards
                curr_estimate_proxy = self._torch_nakano_coeff(i - 1, n_pt - 1, max_n_pt, n_pt) * sorted_rewards[i - 1]
                estimate_proxy = curr_estimate_proxy.sum()
                
                
                    
                prompt_gt_estimates.append(estimate_gt)
                prompt_proxy_estimates.append(estimate_proxy)
            
            selected_actions.append(selected_action)
            selected_gt_rewards.append(selected_gt_reward)
            all_prompts_gt_estimates.append(prompt_gt_estimates)
            all_prompts_proxy_estimates.append(prompt_proxy_estimates)

            
            
        gt_estimates_pt = torch.Tensor(all_prompts_gt_estimates).to(device)
        gt_estimates_proxy = torch.Tensor(all_prompts_proxy_estimates).to(device)
        final_actions = torch.Tensor(selected_actions).to(device)
        final_gt_rewards = torch.Tensor(selected_gt_rewards).to(device)

        estimate_gt_avg = torch.mean(gt_estimates_pt, axis=0)
        estimate_proxy_avg = torch.mean(gt_estimates_proxy, axis=0)

        return estimate_proxy_avg, estimate_gt_avg, final_actions, final_gt_rewards


    def _compute_objective(self, reward_preds, rewards_samples):
        rewards = self._recenter_rewards(reward_preds)
        return rewards
        
    def _win_rate_eval(self, prompts, base_policy, optimal_actions, gt_pref_model):
        actions = base_policy.act(prompts)
        triples = (prompts, optimal_actions, actions)
        prefs, _, _, _ = gt_pref_model.generate_preferences(triples)
        win_rate = prefs.sum() / prefs.shape[0]
        return win_rate.detach().cpu().numpy()

    def _compute_cvar_win_rate(self, r_gt_train, actions_train, prompts, base_policy, gt_pref_model, alpha=0.1):
        alpha_reward = torch.quantile(r_gt_train, alpha)
        mask = r_gt_train <= alpha_reward
        selected_actions = actions_train[mask]
        selected_prompts = prompts[mask]
        base_actions = base_policy.act(selected_prompts)
        triples = (selected_prompts, selected_actions, base_actions)
        prefs, _, _, _ = gt_pref_model.generate_preferences(triples)
        win_rate = prefs.sum() / prefs.shape[0]
        return win_rate.detach().cpu().numpy()

    def _recenter_rewards(self, rewards):
        return rewards - torch.mean(rewards)


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

    def _torch_binom(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1))
    
    def _torch_nakano_coeff(self, a, b, c, d):
        return torch.exp(torch.lgamma(a + 1) + torch.lgamma((c - d) + 1) + torch.lgamma(d + 1) \
                         - torch.lgamma((a - b) + 1) - torch.lgamma(b + 1) - torch.lgamma(c + 1))


