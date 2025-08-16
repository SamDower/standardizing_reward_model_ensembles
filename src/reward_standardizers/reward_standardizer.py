from abc import ABC, abstractmethod

class RewardStandardizer(ABC):

    def __init__(self, config, device, markov_decision_process) -> None:
        self.args = config
        self.device = device
        self.markov_decision_process = markov_decision_process
        self.values = []
        self.norms = []

    @abstractmethod
    def standardize(self, trajectories, t_index, rewards, reward_model):
        pass

    def calculate_values(self, uncertainty_aware_preference_model):
        self.values = []

    def calculate_norms(self, uncertainty_aware_preference_model):
        self.norms = []