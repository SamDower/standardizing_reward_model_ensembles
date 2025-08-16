from trajectory_datasets import TrajectoryDataset
import torch

class Random36NonDetTrajectoryDataset(TrajectoryDataset):

    def _sample_prompts(self, N):
        #return self._generate_random_trajectories(N)
        return (self._generate_random_trajectories(N), self._generate_random_trajectories(N))
    
    def _generate_random_trajectories(self, N):
        trajectories = torch.randint(0, self.mdp.numActions, (N, self.mdp.tMax, 3)).to(self.device)

        trajectories[:,0,0] = torch.remainder(torch.arange(N).to(self.device), self.mdp.numStates)

        for t in range(0, self.mdp.tMax):
            prev_states = trajectories[:,t,0]
            prev_actions = trajectories[:,t,1]
            possible_next_states = self.mdp.possible_next_states_matrix[prev_states, prev_actions] # [N,4]
            possible_next_states_even = torch.cat([possible_next_states[:, 0:1].repeat(1, 7), possible_next_states[:, 1:]], dim=1) # [N,10]
            sampled_next_states = possible_next_states_even[torch.arange(N), torch.randint(0, 10, (N,))]

            trajectories[:,t,2] = sampled_next_states
            if t < self.mdp.tMax-1:
                trajectories[:,t+1,0] = sampled_next_states
    
        return trajectories.to(torch.int64)