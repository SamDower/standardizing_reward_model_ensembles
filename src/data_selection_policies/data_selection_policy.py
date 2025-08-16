from abc import ABC, abstractmethod

class DataSelectionPolicy(ABC):
    def __init__(self, args):
        self.args = args
        
    @abstractmethod
    def select_data(self, dataset, preference_model, batch_size):
        """
        Selects data based on the given policy and reward model.

        Args:
            dataset (list): The dataset to select data from.
            preference_model (object): The reward model used for data selection.

        Returns:
            list: List of tuples, each containing (prompt, action_1, action_2).
        """
        pass
