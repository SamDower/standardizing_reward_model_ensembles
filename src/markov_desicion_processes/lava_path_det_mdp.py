from markov_desicion_processes import MDP
from reward_standardizers import GTStarcRewardStandardizer
import torch

class LavaPathDetMDP(MDP):

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.gridHeight = 6
        self.gridWidth = 6
        self.numActions = 4
        self.numStates = 36
        # self.rewards = torch.tensor([
        #     [-5, -2, -5, -5, -3, -5],
        #     [-3, -2, -0, -1,  -3, -5],
        #     [-.1,  -.1,  -.1,  -.1,  -.1,  -.1 ],
        #     [-.1,  -5, -.1,  -5, -5, -5],
        #     [-.1,  -5, -.1,  -5, -3, -5],
        #     [-.1,  -5, -.1,  -.1,  10, -2]
        # ]).reshape(36).to(device)
        self.rewards = torch.tensor([
            [-50, -20, -50, -50, -30, -50],
            [-30, -20, -50,  -1,  -30, -50],
            [-1,  -1,  -1,  -1,  -1,  -1 ],
            [-1,  -50, -1,  -50, -50, -50],
            [-1,  -50, -1,  -50, -30, -50],
            [-1,  -50, -1,  -1,  100, -20]
        ]).reshape(36).to(device)
        # self.terminatingStates = torch.zeros(36).to(device)
        # self.terminatingStates[34] = 1
        self.tMax = 16
        self.next_state_matrix = self._generate_next_state_matrix()
        self.reward_table = self._generate_reward_table()
        self.stand_reward_table = self._generate_stand_reward_table()
        
    def get_reward(self, sa):
        return self.reward_table[sa[0]][sa[1]]
        return self.rewards[self.next_state_matrix[sas[0]][sas[1]]] * (1 - self.terminatingStates[sas[0]])
        nextX, nextY = self._n_to_xy(self.get_next_state(sa))
        return self.rewards[nextX][nextY]
    
    def get_next_state(self, sa):
        return self.next_state_matrix[sa[0]][sa[1]]
    
    def get_initial_state(self):
        return 30
    
    def _xy_to_n(self, xy):
        return xy[0] * self.gridWidth + xy[1]

    def _n_to_xy(self, n):
        return (n // self.gridWidth, n % self.gridWidth)
    
    def _generate_next_state_matrix(self):
        next_state_matrix = torch.zeros(self.numStates, self.numActions, dtype=torch.int).to(self.device)
        for s in range(self.numStates):
            for a in range(self.numActions):
                # if self.terminatingStates[s] == 1:
                #     next_state_matrix[s][a] = s
                #else:
                x, y = self._n_to_xy(s)
                if a == 1 and y < self.gridHeight-1:
                    next_state_matrix[s][a] = self._xy_to_n((x, y+1)) # Down
                elif a == 0 and y > 0:
                    next_state_matrix[s][a] = self._xy_to_n((x, y-1)) # Up
                elif a == 2 and x > 0:
                    next_state_matrix[s][a] = self._xy_to_n((x-1, y)) # Left
                elif a == 3 and x < self.gridWidth-1:
                    next_state_matrix[s][a] = self._xy_to_n((x+1, y)) # Right
                else:
                    next_state_matrix[s][a] = s
        return next_state_matrix
    
    def _generate_reward_table(self):
        reward_table = torch.zeros(self.numStates, self.numActions).to(self.device)
        for s in range(36):
            for a in range(4):
                reward_table[s][a] = self.rewards[self.next_state_matrix[s][a]] #* (1 - self.terminatingStates[s])
        return reward_table

    def _generate_stand_reward_table(self):
        stand_reward_table = torch.zeros(self.numStates, self.numActions, 16).to(self.device)
        standardizer = GTStarcRewardStandardizer(self.device, self)
        standardizer.calculate_values()
        standardizer.calculate_norms()
        for s in range(36):
            for a in range(4):
                sa = torch.tensor([s,a]).reshape(1,2).to(self.device)
                for t in range(16):
                    stand_reward_table[s][a][t] = standardizer.standardize(sa, t)
        return stand_reward_table