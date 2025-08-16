from policies import Policy
import torch

class UniformPolicy(Policy):
    def __init__(self, config, device):
        """
        Initializes the UniformPolicy with an action space for sampling.

        Args:
            action_space (tuple): A tuple of two integers representing the lower and upper bounds of the sampling interval.
        """
        self.config = config
        self.device = device

    def act(self, context: torch.Tensor) -> torch.Tensor:
        """
        Uniformly samples a number from the action space for each row in the context array.

        Args:
            context (np.ndarray): A NumPy array with shape (n_rows, n_features).

        Returns:
            np.ndarray: An array of uniformly sampled numbers, one per row in the context.
        """
        n_rows = context.shape[0]
        samples = torch.FloatTensor(n_rows).uniform_(self.config['action_space_lb'], self.config['action_space_ub']).to(context.device)
        return samples
