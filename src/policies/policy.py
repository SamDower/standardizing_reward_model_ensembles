from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def act(self, context):
        """
        Returns an policy action given the context.

        Args:
            context (list): List of data points for evaluation.

        Returns:
            policy action.
        """
        pass

