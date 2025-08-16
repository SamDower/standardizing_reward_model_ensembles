from markov_desicion_processes import MDP
import torch

class LavaPathMDP(MDP):

    def __init__(self, device):
        self.gridHeight = 2
        self.gridWidth = 3
        self.numActions = 4
        self.numStates = 36
        self.rewards = torch.tensor([
            [-5, -2, -5, -5, -3, -5],
            [-3, -2, -0, -1,  -3, -5],
            [-.1,  -.1,  -.1,  -.1,  -.1,  -.1 ],
            [-.1,  -5, -.1,  -5, -5, -5],
            [-.1,  -5, -.1,  -5, -3, -5],
            [-.1,  -5, -.1,  -.1,  10, -2]
        ]).reshape(36).to(device)
        self.terminatingStates = torch.zeros(36).to(device)
        self.terminatingStates[34] = 1
        self.tMax = 16
        self.device = device
        
    def get_reward(self, sa):
        return self.rewards[self.get_next_state(sa)] * (1 - self.terminatingStates[sa[0]])
        nextX, nextY = self._n_to_xy(self.get_next_state(sa))
        return self.rewards[nextX][nextY]
    
    def get_next_state(self, sa):
        
        if self.terminatingStates[sa[0]] == 1:
            return sa[0]
        
        x, y = self._n_to_xy(sa[0])
        if sa[1] == 0 and y < self.gridHeight-1:
            return self._xy_to_n((x, y+1))
        if sa[1] == 1 and y > 0:
            return self._xy_to_n((x, y-1))
        if sa[1] == 2 and x > 0:
            return self._xy_to_n((x-1, y))
        if sa[1] == 3 and x < self.gridWidth-1:
            return self._xy_to_n((x+1, y))
        return sa[0]
    
    def get_initial_state(self):
        return 30
    
    def _xy_to_n(self, xy):
        return xy[0] * self.gridWidth + xy[1]

    def _n_to_xy(self, n):
        return (n // self.gridWidth, n % self.gridWidth)