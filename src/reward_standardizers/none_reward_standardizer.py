from reward_standardizers import RewardStandardizer

class NoneRewardStandardizer(RewardStandardizer):

    def __init__(self, config, device, markov_decision_process) -> None:
        self.args = config
        self.device = device
        self.markov_decision_process = markov_decision_process
        self.policies = []
        self.norms = []

    def standardize(self, state_action_inputs, t_index, rewards, reward_model, nondet=False):
        if not nondet:
            return rewards
        
        else:
            states = state_action_inputs[:, 0]
            actions = state_action_inputs[:, 1]
            next_states = state_action_inputs[:, 2]
            return rewards[states, actions, next_states]

