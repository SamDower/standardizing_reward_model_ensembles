from abc import ABC, abstractmethod


class TrajectoryDataset(ABC):
    def __init__(self, config, mdp, device=None):
        self.config = config
        self.mdp = mdp
        self.device = device
        self.samples = None

    def generate_dataset(self, num_samples):
        self.samples = self._sample_prompts(num_samples)

    def update_data(self, new_data):
        self.pool = new_data

    @abstractmethod
    def _sample_prompts(self, N):
        pass

    def retrieve_data(self, prompt_ids, answer_ids):
        return prompt_ids, answer_ids
