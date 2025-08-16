from abc import ABC, abstractmethod
import torch

class PreferenceModel(ABC):
    def __init__(self, model):
        """
        Initializes the GroundTruthRewardModel with a given PreferenceModel.

        Args:
            preference_model (object): The preference model to use.
        """
        self.model = model

    @abstractmethod
    def generate_preferences(self, prompt_response_triple, mode, policy_for_starc=None, override_standardizer=False):
        """
        Generates preferences based on the given prompt-response triple.

        Args:
            prompt_response_triple (tuple): A tuple containing (prompt, action_1, action_2).

        """
        prompt, action_1, action_2 = prompt_response_triple
        rewards_first = self.generate_reward((prompt, action_1))
        rewards_second = self.generate_reward((prompt, action_2))
        # Computing y_1 >= y_2. If label is 1, then first is preferred over second (p >= 0.5)
        pref_probs = torch.sigmoid(rewards_first - rewards_second).to(device=prompt.device)
        preferences = torch.Tensor(rewards_first >= rewards_second).to(dtype=torch.int, device=prompt.device)
        return preferences, pref_probs, rewards_first, rewards_second

    @abstractmethod
    def generate_reward(self, prompt_action_pair, features=True, policy_for_starc=None, override_standardizer=False):
        """
        Generates a reward based on the given prompt-action pair.

        Args:
            prompt_action_pair (tuple): A tuple containing (prompt, action_1).

        Returns:
            float: Reward value for the specified action.
        """
        prompt, action_1 = prompt_action_pair
        # Implement your reward generation logic here
        # ...
        pass

    def eval(self, train_data, eval_data, gt_preference_model, policy_for_starc=None):
        """
        Evals the preference model using the provided data.

        Args:
            train_data (list): List of preference data used in training (e.g., prompt-response pairs).
            eval_data (list): List of preference data NOT used in training (e.g., prompt-response pairs).
            gt_preference_model (PreferenceModel): Ground truth preference model.

        Returns:
            eval_dict (dict): Dictionary with reward model metrics.
        """
        
        # Train
        pref_model_train, pref_probs_model_train, rw_model_first_train, rw_model_second_train = self.generate_preferences(train_data)
        pref_gt_train, _, rw_gt_first_train, rw_gt_second_train = gt_preference_model.generate_preferences(train_data)

        # Test
        pref_model_test, pref_probs_model_test, rw_model_first_test, rw_model_second_test = self.generate_preferences(eval_data)
        pref_gt_test, _, rw_gt_first_test, rw_gt_second_test = gt_preference_model.generate_preferences(eval_data)

        # Compute LogLikelihoods
        ll_train = self._compute_log_likelihood(pref_gt_train, pref_probs_model_train).mean().detach().cpu().numpy()
        ll_test = self._compute_log_likelihood(pref_gt_test, pref_probs_model_test).mean().detach().cpu().numpy()

        # Compute Accuracy
        acc_train = self._compute_acc(pref_gt_train, pref_model_train)
        acc_test = self._compute_acc(pref_gt_test, pref_model_test)

        # Organize Tuples
        rewards_train = self._build_rw_tuples(train_data, rw_model_first_train, rw_model_second_train, rw_gt_first_train, rw_gt_second_train)
        rewards_test = self._build_rw_tuples(eval_data, rw_model_first_test, rw_model_second_test, rw_gt_first_test, rw_gt_second_test)

        eval_dict = {
            'll_train': ll_train,
            'll_test': ll_test,
            'acc_train': acc_train,
            'acc_test': acc_test,
            'rewards_train': rewards_train,
            'rewards_test': rewards_test
        }

        return eval_dict

    def _compute_log_likelihood(self, labels, probs):
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        return labels * torch.log(probs) + (1.0 - labels) * torch.log(1.0 - probs)
    
    def _compute_acc(self, labels, preds):
        acc = (preds == labels).sum().item() / len(labels)
        return acc

    def _build_rw_tuples(self, train_data, rw_model_first, rw_model_second, rw_gt_first, rw_gt_second):
        # if len(train_data) == 1:
        #     prompts, a1, a2 = train_data[0], torch.zeros(train_data[0].shape), torch.ones(train_data[0].shape)
        # else:
        #     prompts, a1, a2 = train_data
            
        # final_prompts = torch.cat((prompts, prompts), axis=0)
        # actions = torch.cat((a1, a2), axis=0)
        rws_model = torch.cat((rw_model_first, rw_model_second), axis=0)
        rws_gt = torch.cat((rw_gt_first, rw_gt_second), axis=0)
        return (train_data[0], train_data[1], rws_model, rws_gt)
        return (final_prompts, actions, rws_model, rws_gt)
    
    def _build_prompt_action_pairs(self, prompt_response_triple):
        prompt, action_1, action_2 = prompt_response_triple
        return (prompt, action_1), (prompt, action_2)
