from abc import ABC, abstractmethod
import torch
import numpy as np

class MDP(ABC):
    def __init__(self, args, device):
        """
        """
        self.args = args
        self.reward_table = []

    @abstractmethod
    def get_reward(self, sa):
        pass

    @abstractmethod
    def get_next_state(self, sa):
        pass

    @abstractmethod
    def get_initial_state(self):
        pass

    def flip_bits(self, tensor, p):
        # Create a random tensor with the same shape, values between 0 and 1
        random_tensor = torch.rand_like(tensor, dtype=torch.float).to(self.device)
        # Decide where to flip: where random values are less than p
        flip_mask = random_tensor < p
        # Flip bits using XOR (1 - bit is equivalent to bit ^ 1)
        flipped_tensor = tensor ^ flip_mask.to(tensor.dtype)
        return flipped_tensor

    def generate_gt_preferences(self, trajectory_pairs, add_aleatoric=False):
        trajectories_first, trajectories_second = trajectory_pairs
        rewards_first = self._generate_gt_trajectory_reward(trajectories_first)
        rewards_second = self._generate_gt_trajectory_reward(trajectories_second)
        # Computing y_1 >= y_2. If label is 1, then first is preferred over second (p >= 0.5)
        pref_probs = torch.sigmoid(rewards_first - rewards_second).to(device=self.device)
        preferences = torch.Tensor(rewards_first >= rewards_second).to(dtype=torch.int, device=self.device)
        if add_aleatoric:
            preferences = self.flip_bits(preferences, self.args['flip_preferences_prob'])
        return preferences, pref_probs, rewards_first, rewards_second
    
    def _generate_gt_trajectory_reward(self, trajectories):
        N = trajectories.shape[0]
        rewards = torch.zeros(N).to(self.device)
        for t in range(trajectories.shape[1]):
            states = trajectories[:,t,0]
            actions = trajectories[:,t,1]
            rewards += self.reward_table[states, actions] # shape [N]
        return rewards


        rewards = torch.zeros(trajectories.shape[0], trajectories.shape[1]).to(self.device)
        for i in range(rewards.shape[0]):
            for t in range(rewards.shape[1]):
                rewards[i][t] = self.get_reward(trajectories[i][t])
        return rewards.sum(axis=1)

        traj2 = trajectories.cpu()
        iterable = (tuple(self.get_reward(traj2[i][t]) for t in range(traj2.shape[1])) for i in range(traj2.shape[0]))
        individual_rewards = torch.from_numpy(np.fromiter(iterable, dtype=np.dtype((int, traj2.shape[1])), count=traj2.shape[0])).to(self.device)
        return individual_rewards.sum(axis=1)
    