from policies import Policy
import torch
import numpy as np
import torch.nn.functional as F

class BoltzmannPolicy(Policy):
    def __init__(self, config, markov_decision_process, device):
        """
        Initializes the UniformPolicy with an action space for sampling.

        Args:
            action_space (tuple): A tuple of two integers representing the lower and upper bounds of the sampling interval.
        """
        self.config = config
        self.markov_decision_process = markov_decision_process
        self.device = device
        self.Q_table = torch.zeros((markov_decision_process.numStates, markov_decision_process.numActions)).to(device)
        self.values = torch.zeros(36)
        self.beta = 1

    def act(self, s: torch.Tensor) -> torch.Tensor:
        """
        Uniformly samples a number from the action space for each row in the context array.

        Args:
            context (np.ndarray): A NumPy array with shape (n_rows, n_features).

        Returns:
            np.ndarray: An array of uniformly sampled numbers, one per row in the context.
        """

        return torch.argmax(self.Q_table[s,:])

    def act_boltzmann(self, s: torch.Tensor) -> torch.Tensor:
        pre_exp_values = self.beta * self.Q_table[s,:]
        pre_exp_values = pre_exp_values - torch.max(pre_exp_values)
        exp_values = torch.exp(pre_exp_values)
        probs = exp_values / exp_values.sum()
        dist=torch.distributions.categorical.Categorical(probs=probs)
        return dist.sample()
    
    def act_out_trajectories(self, N):
        trajectories = torch.zeros(N, self.markov_decision_process.tMax, 2, dtype=torch.int64).to(self.device)
        for i in range(N):
            trajectories[i][0][0] = self.markov_decision_process.get_initial_state()
            trajectories[i][0][1] = self.act(trajectories[i][0][0])
            for t in range(1, self.markov_decision_process.tMax):
                trajectories[i][t][0] = self.markov_decision_process.get_next_state(trajectories[i][t-1])
                trajectories[i][t][1] = self.act(trajectories[i][t][0])
        return trajectories
    
    def act_boltzmann_out_trajectories(self, N):
        trajectories = torch.zeros(N, self.markov_decision_process.tMax, 2, dtype=torch.int64).to(self.device)
        for i in range(N):
            trajectories[i][0][0] = self.markov_decision_process.get_initial_state()
            trajectories[i][0][1] = self.act_boltzmann(trajectories[i][0][0])
            for t in range(1, self.markov_decision_process.tMax):
                trajectories[i][t][0] = self.markov_decision_process.get_next_state(trajectories[i][t-1])
                trajectories[i][t][1] = self.act_boltzmann(trajectories[i][t][0])
        return trajectories
    
    def train_sarsa_on_mdp(self):
        print("Generating reward table")
        reward_table = torch.zeros(36, 4).to(self.device)
        for s in range(36):
            for a in range(4):
                reward_table[s][a] = self.markov_decision_process.get_reward(torch.tensor([s, a]).to(self.device))
        self.train_sarsa(reward_table)

    def train_sarsa_on_preference_model(self, preference_model):
        print("Generating reward table")
        reward_table = torch.zeros(36, 4).to(self.device)
        for s in range(36):
            for a in range(4):
                sa = torch.tensor([s, a]).to(self.device).reshape(1, 2)
                reward, _ = preference_model.generate_state_actions_reward(sa, use_standardizer=False)
                reward_table[s][a] = reward
        self.train_sarsa(reward_table)


    def train_sarsa_on_model(self, model):
        print("Generating reward table")
        reward_table = torch.zeros(36, 4).to(self.device)
        for s in range(36):
            state_actions = torch.tensor([[s,0],[s,1],[s,2],[s,3]]).to(self.device)
            one_hot_inputs = torch.cat((F.one_hot(state_actions[:,0], num_classes=36), F.one_hot(state_actions[:,1], num_classes=4)), dim = 1).float()
            rewards = model(one_hot_inputs)
            for a in range(4):
                reward_table[s][a] = rewards[a]
        self.train_sarsa(reward_table)

    def _act_exploratorily(self, s, exploration_proba) -> torch.Tensor:
        if np.random.uniform(0,1) < exploration_proba:
            return np.random.randint(self.markov_decision_process.numActions)
        else:
            return torch.argmax(self.Q_table[s,:])
        
    def train_sarsa(self, reward_table):
        n_episodes = 5000
        max_iter_episode = 20
        exploration_proba = 0.1
        gamma = 0.99
        lr = 0.1

        rewards_per_episode = []
        for e in range(n_episodes):
            
            if e % 500 == 0:
                print(f"Running episode {e}")
            
            current_state = self.markov_decision_process.get_initial_state()
            current_action = self._act_exploratorily(current_state, exploration_proba)
            total_episode_reward = 0
            
            for i in range(max_iter_episode): 
                
                sa = torch.tensor([current_state, current_action]).to(self.device)
                reward = reward_table[sa[0]][sa[1]]
                next_state = self.markov_decision_process.get_next_state(sa)
                next_action = self._act_exploratorily(next_state, exploration_proba)
                
                self.Q_table[current_state, current_action] = (1-lr) * self.Q_table[current_state, current_action] +lr*(reward + gamma*self.Q_table[next_state,next_action])
                # self.Q_table[current_state, action] = (1-lr) * self.Q_table[current_state, action] +lr*(reward + gamma*self.Q_table[next_state,self._act_exploratorily(next_state, exploration_proba)])
                total_episode_reward = total_episode_reward + reward

                current_state = next_state
                current_action = next_action

            #We update the exploration proba using exponential decay formula 
            #exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_episode_reward)
        
        print("Mean reward per thousand episodes")
        for i in range(50):
            print(f"{(i+1)*100}: mean espiode reward: {torch.mean(torch.tensor(rewards_per_episode[100*i:100*(i+1)]).to(self.device))}")
        
        print(torch.reshape(torch.max(self.Q_table, 1).values, (6,6)))




    # def train_sarsa(self, reward_table):
    #     n_episodes = 5000
    #     max_iter_episode = 20
    #     exploration_proba = 1
    #     exploration_decreasing_decay = 0.001
    #     min_exploration_proba = 0.01
    #     gamma = 0.99
    #     lr = 0.1

    #     rewards_per_episode = []
    #     for e in range(n_episodes):
            
    #         if e % 500 == 0:
    #             print(f"Running episode {e}")
            
    #         current_state = self.markov_decision_process.get_initial_state()
    #         total_episode_reward = 0
            
    #         for i in range(max_iter_episode): 
                
    #             action = self._act_exploratorily(current_state, exploration_proba)
    #             sa = torch.tensor([current_state, action]).to(self.device)
    #             next_state = self.markov_decision_process.get_next_state(sa)
    #             reward = reward_table[sa[0]][sa[1]]
                
    #             self.Q_table[current_state, action] = (1-lr) * self.Q_table[current_state, action] +lr*(reward + gamma*self.Q_table[next_state,self.act(next_state)])
    #             # self.Q_table[current_state, action] = (1-lr) * self.Q_table[current_state, action] +lr*(reward + gamma*self.Q_table[next_state,self._act_exploratorily(next_state, exploration_proba)])
    #             total_episode_reward = total_episode_reward + reward

    #             current_state = next_state

    #         #We update the exploration proba using exponential decay formula 
    #         exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
    #         rewards_per_episode.append(total_episode_reward)
        
    #     print("Mean reward per thousand episodes")
    #     for i in range(50):
    #         print(f"{(i+1)*100}: mean espiode reward: {torch.mean(torch.tensor(rewards_per_episode[100*i:100*(i+1)]).to(self.device))}")
        
    #     print(torch.reshape(torch.max(self.Q_table, 1).values, (6,6)))
