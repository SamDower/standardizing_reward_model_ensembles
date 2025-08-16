from abc import ABC, abstractmethod

class Trainer(ABC):
    @abstractmethod
    def train_and_evaluate(self, train_prompts, eval_prompts, policy, preference_model, gt_preference_model, full_retrain=True, kl_budget=2.0):
        """
        Trains the policy using the given data and reward model.

        Args:
            data (list): List of data points for training.
            reward_model (object): The reward model used for training.

        Returns:
            None
        """
        pass
