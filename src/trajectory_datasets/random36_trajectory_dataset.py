from trajectory_datasets import TrajectoryDataset
import torch

class Random36TrajectoryDataset(TrajectoryDataset):

    def _sample_prompts(self, N):
        #return self._generate_random_trajectories(N)
        return (self._generate_random_trajectories(N), self._generate_random_trajectories(N))
    
    def _generate_random_trajectories(self, N):
        trajectories = torch.randint(0, self.mdp.numActions, (N, self.mdp.tMax, 2)).to(self.device)

        trajectories[:,0,0] = torch.remainder(torch.arange(N).to(self.device), self.mdp.numStates)
        for t in range(1, self.mdp.tMax):
            prev_states = trajectories[:,t-1,0]
            prev_actions = trajectories[:,t-1,1]
            trajectories[:,t,0] = self.mdp.next_state_matrix[prev_states, prev_actions]

        return trajectories.to(torch.int64)
    