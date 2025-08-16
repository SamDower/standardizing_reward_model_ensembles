from trajectory_datasets import TrajectoryDataset
import torch

class ExpBiasedTrajectoryDataset(TrajectoryDataset):

    def __init__(self, config, mdp, device=None):
        self.config = config
        self.mdp = mdp
        self.device = device
        self.samples = None

        big_p = 0.55
        probs = torch.tensor([big_p/2, (1-big_p)/2, (1-big_p)/2, big_p/2])
        self.action_dist = torch.distributions.categorical.Categorical(probs=probs)

        # pre_exp_values = torch.linspace(0, 3, steps=4) * 2
        # pre_exp_values = pre_exp_values - torch.max(pre_exp_values)
        # exp_values = torch.exp(pre_exp_values)
        # probs = exp_values / exp_values.sum()
        # self.action_dist = torch.distributions.categorical.Categorical(probs=probs)

    def _sample_prompts(self, N):
        #return self._generate_random_trajectories(N)
        return (self._generate_random_trajectories(N), self._generate_random_trajectories(N))
    
    def _generate_random_trajectories(self, N):
        trajectories = torch.zeros(N, self.mdp.tMax, 3).to(torch.int64).to(self.device)
    
        for t in range(self.mdp.tMax):
            for i in range(N):
                if t == 0:
                    trajectories[i][t][0] = self.mdp.get_initial_state()
                else:
                    trajectories[i][t][0] = self.mdp.get_next_state(trajectories[i][t-1][:2])
                trajectories[i][t][1] = self.action_dist.sample()
                trajectories[i][t][2] = self.mdp.get_next_state(trajectories[i][t][:2])

        return trajectories
    