import torch
import numpy as np
import gc

class UncertaintyAwareRewardModel:
    def train(self, data, full_retrain=True):
        """
        Trains the preference model using the provided data.

        Args:
            data (list): List of preference data (e.g., prompt-response pairs).

        Returns:
            None
        """
        # Implement your training logic here
        # ...
        pass

    def featurize(self, prompt_action_pair):
        pass

    def generate_preferences(self, trajectory_pairs, use_standardizer=True, use_table=False, use_set_mag=False):
        """
        Generates preferences based on the given prompt-response triple.

        Args:
            trajectory_pairs: Tuple of trajectories (trajectories_1, trajectories_2)

        """
        trajectories_1, trajectories_2 = trajectory_pairs
        rewards_first_mean, rewards_first_samples = self.generate_reward(trajectories_1, use_standardizer=use_standardizer, use_table=use_table)
        rewards_second_mean, rewards_second_samples = self.generate_reward(trajectories_2, use_standardizer=use_standardizer, use_table=use_table)
        
        if use_set_mag:
            scale = self.args['set_reward_magnitude'] / self.mean_stand_reward_magnitude
            rewards_first_mean = rewards_first_mean * scale
            rewards_first_samples = rewards_first_samples * scale
            rewards_second_mean = rewards_second_mean * scale
            rewards_second_samples = rewards_second_samples * scale

        # Compute mean preference probabilities
        # Computing y_1 >= y_2. If label is 1, then first is preferred over second (p >= 0.5)
        pref_probs = torch.sigmoid(rewards_first_mean - rewards_second_mean).to(device=rewards_first_mean.device)
        preferences = torch.Tensor(rewards_first_mean >= rewards_second_mean).to(dtype=torch.int, device=rewards_first_mean.device) # 0 means first is preferred; 1 otherwise

        # Compute distribution over preference probabilities 
        pref_probs_samples = torch.sigmoid(rewards_first_samples - rewards_second_samples).to(device=rewards_first_mean.device)
        return preferences, pref_probs, rewards_first_samples, rewards_second_samples, pref_probs_samples

    def eval(self, eval_data, markov_decision_process):
        """
        Evals the preference model using the provided data.

        Args:
            train_data: Tuple of trajectories (trajectories_1, trajectories_2)
            eval_data: Tuple of trajectories (trajectories_1, trajectories_2)
            markov_decision_process (MDP): Ground truth preference model.

        Returns:
            eval_dict (dict): Dictionary with reward model metrics.
        """

        with torch.no_grad():
            # Train
            # pref_model_train, pref_probs_model_train, rw_model_first_train, rw_model_second_train, _ = self.generate_preferences(train_data)
            # pref_gt_train, _, rw_gt_first_train, rw_gt_second_train = markov_decision_process.generate_gt_preferences(train_data)

            # Test
            # pref_model_test, pref_probs_model_test, rw_model_first_test, rw_model_second_test, _ = self.generate_preferences(eval_data)
            # pref_gt_test, _, rw_gt_first_test, rw_gt_second_test = markov_decision_process.generate_gt_preferences(eval_data)

            # Test raw (no standardiser)
            pref_model_test_stand, pref_probs_model_test_stand, rw_model_first_test_stand, rw_model_second_test_stand, _ = self.generate_preferences(eval_data, use_standardizer=True, use_table=True)
            pref_model_test_raw, pref_probs_model_test_raw, rw_model_first_test_raw, rw_model_second_test_raw, _ = self.generate_preferences(eval_data, use_standardizer=False, use_table=True)
            
            pref_gt_test, _, rw_gt_first_test, rw_gt_second_test = markov_decision_process.generate_gt_preferences(eval_data)

            # Compute LogLikelihoods
            #ll_train = self._compute_log_likelihood(pref_gt_train, pref_probs_model_train).mean().detach().cpu().numpy()
            #ll_test = self._compute_log_likelihood(pref_gt_test, pref_probs_model_test_raw).mean().detach().cpu().numpy()

            # Compute Accuracy
            #acc_train = self._compute_acc(pref_gt_train, pref_model_train)
            acc_test_stand = self._compute_acc(pref_gt_test, pref_model_test_stand)
            acc_test_raw = self._compute_acc(pref_gt_test, pref_model_test_raw)

            # Organize Tuples
            #rewards_train = self._build_rw_tuples(train_data, rw_model_first_train, rw_model_second_train, rw_gt_first_train, rw_gt_second_train)
            #rewards_test = self._build_rw_tuples(eval_data, rw_model_first_test, rw_model_second_test, rw_gt_first_test, rw_gt_second_test)
            #rewards_test = self._build_rw_tuples(eval_data, rw_model_first_test_raw, rw_model_second_test_raw, rw_gt_first_test, rw_gt_second_test)
            
            # Reward tables
            raw_reward_table = torch.mean(self.raw_reward_table, dim=0)
            raw_reward_variances_table = self.raw_reward_variances_table
            standardized_reward_table = torch.mean(self.stand_reward_table, dim=0)
            standardized_reward_variances_table = self.stand_reward_variances_table
            # raw_reward_table = torch.zeros(36, 4).to(self.device)
            # raw_reward_variances_table = torch.zeros(36, 4).to(self.device)
            # standardized_reward_table = torch.zeros(36, 4, 16).to(self.device)
            # standardized_reward_variances_table = torch.zeros(36, 4, 16).to(self.device)
            # for s in range(36):
            #     for a in range(4):
            #         sa = torch.tensor([s, a]).to(self.device).reshape(1, 2)
            #         raw_reward, all_raw_rewards = self.generate_state_actions_reward(sa, _, use_standardizer=False)
            #         raw_reward_table[s][a] = raw_reward
            #         raw_reward_variances_table[s][a] = all_raw_rewards.var()
            #         for t in range(16):
            #             stand_reward, all_stand_rewards = self.generate_state_actions_reward(sa, t, use_standardizer=True)
            #             standardized_reward_table[s][a] = stand_reward
            #             standardized_reward_variances_table[s][a] = all_stand_rewards.var()

            # STARC distances
            #starc_distance = torch.sum(torch.abs(markov_decision_process.stand_reward_table - standardized_reward_table)).item()

            # Dataset Statistics
            data_stats = self._build_data_stats()

            eval_dict = {
                'll_train': None, #ll_train,
                'll_test': None, #ll_test,
                'acc_train': None, #acc_train,
                'acc_test_raw': acc_test_raw,
                'acc_test_stand': acc_test_stand,
                'rewards_train': None, #rewards_train,
                'rewards_test': None, #rewards_test,
                'data_stats': data_stats,
                'full_data': self.full_data[0].cpu(),
                'raw_reward_table': raw_reward_table.cpu(),
                'raw_reward_variances_table': raw_reward_variances_table.cpu(),
                'standardized_reward_table': standardized_reward_table.cpu(),
                'standardized_reward_variances_table': standardized_reward_variances_table.cpu(),
                'starc_distance': None #starc_distance,
            }

            del pref_gt_test, #pref_probs_model_test, rw_model_first_test, rw_model_second_test
            del rw_gt_first_test, rw_gt_second_test
        
        gc.collect()
        torch.cuda.empty_cache()

        return eval_dict
    
    def _build_data_stats(self):
        trajectories = self.full_data[0]
        hist_s, bins_s = np.histogram(torch.flatten(trajectories[:,:,0]).cpu().numpy(), bins=36, range=(0,36))
        hist_a, bins_a = np.histogram(torch.flatten(trajectories[:,:,1]).cpu().numpy(), bins=4, range=(0,4))
        return {
            'states': (hist_s, bins_s),
            'actions': (hist_a, bins_a)
        }

    def _compute_acc(self, labels, preds):
        acc = (preds == labels).sum().item() / len(labels)
        return acc