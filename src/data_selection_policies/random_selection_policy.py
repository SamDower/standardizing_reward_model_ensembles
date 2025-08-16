from data_selection_policies import DataSelectionPolicy

import torch

class RandomSelectionPolicy(DataSelectionPolicy):
    def select_data(self, dataset, preference_model, batch_size):
        """
        Randomly selects N points from the given dataset.

        Args:
            dataset (list): The dataset to select data from.

        Returns:
            list: List of randomly selected data points.
        """

        # Randomly select N points
        data = dataset.samples
        permuted_elements = torch.randperm(data[0].shape[0])
        ids = permuted_elements[:batch_size]

        return (data[0][ids], data[1][ids])

        # Randomly select N points
        data = dataset.samples
        permuted_elements_1 = torch.randperm(data.shape[0])
        permuted_elements_2 = torch.randperm(data.shape[0])
        ids_1 = permuted_elements_1[:batch_size]
        ids_2 = permuted_elements_2[:batch_size]

        return (data[ids_1], data[ids_2])

        remmaining_ids = permuted_elements[batch_size:]
        selected_data = data[ids]
        remaining_data = data[remmaining_ids]

        action_one = policy.act(selected_data)
        action_two = policy.act(selected_data)
        triplets = (selected_data, action_one, action_two)
        dataset.update_data(remaining_data)

        del permuted_elements, ids, remmaining_ids, remaining_data
        return #selected_data, triplets, dataset
