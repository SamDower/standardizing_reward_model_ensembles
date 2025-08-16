from sklearn.model_selection import train_test_split
from modules import SimpleMLP
from preference_models import UncertaintyAwarePreferenceModel
#from reward_standardizers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from copy import deepcopy

class EnsemblePreferenceModel(UncertaintyAwarePreferenceModel):
    def __init__(self, device, reward_standardizer, **kwargs):
        self.args = kwargs
        self.device = device
        self.reward_standardizer = reward_standardizer
        self.models = []
        for i in range(self.args['ensemble_size']):
            self.models.append(SimpleMLP(**self.args['model']).to(device))
        self.standardize = kwargs['standardize_features']
        self.full_data = None
        self._data_min = None
        self._data_max = None
        self.raw_reward_table = torch.zeros(self.args['ensemble_size'], 36, 4).to(self.device)
        self.raw_reward_variances_table = torch.zeros(36, 4).to(self.device)
        self.stand_reward_table = torch.zeros(self.args['ensemble_size'], 36, 4, 16).to(self.device)
        self.stand_reward_variances_table = torch.zeros(36, 4, 16).to(self.device)
        self.mean_stand_reward_magnitude = 1
    
    def generate_raw_reward_table(self):
        for s in range(36):
            for a in range(4):
                sa = torch.tensor([s, a]).to(self.device).reshape(1, 2)
                _, all_raw_rewards = self.generate_state_actions_reward(sa, -1, use_standardizer=False)
                self.raw_reward_table[:,s,a] = all_raw_rewards
                self.raw_reward_variances_table[s][a] = all_raw_rewards.var()

    def generate_stand_reward_table(self):
        for s in range(36):
            for a in range(4):
                sa = torch.tensor([s, a]).to(self.device).reshape(1, 2)
                for t in range(16):
                    _, all_stand_rewards = self.generate_state_actions_reward(sa, t, use_standardizer=True)
                    self.stand_reward_table[:,s,a,t] = all_stand_rewards
                    self.stand_reward_variances_table[s][a][t] = all_stand_rewards.var()
        self.mean_stand_reward_magnitude = torch.sum(torch.abs(self.stand_reward_table.mean(dim=0)))
    
    def generate_preferences(self, trajectory_pairs, use_standardizer=True, use_table=False, use_set_mag=False):
        return super().generate_preferences(trajectory_pairs, use_standardizer=use_standardizer, use_table=use_table, use_set_mag=use_set_mag)
    
    # def generate_sum_of_variances(self, trajectories, use_standardizer=True):
        
    #     realizations = []
    #     for i in range(self.args['ensemble_size']): 
    #         # input is shape [N, tMax, 2]
    #         rewards = torch.zeros(trajectories.shape[0], trajectories.shape[1], 1).to(self.device) # [N, tMax]
    #         for t in range(trajectories.shape[1]):
    #             timestep = trajectories[:,t,:]
    #             rewards_timestep = self.compute_state_action_rewards_from_model(timestep, t, i, use_standardizer=use_standardizer)
    #             rewards[:,t] = rewards_timestep
    #         realizations.append(rewards)

    #     all_preds = torch.cat(realizations, dim=2)
    #     variances = torch.var(all_preds, dim=2)
    #     sum_of_variances = torch.sum(variances, dim=1)
    #     del all_preds, variances, realizations
    #     return sum_of_variances


    def generate_reward(self, trajectories, use_standardizer=True, use_table=False):
        """
        Generates a reward based on the given prompt-action pair.

        Args:
            prompt_action_pair (tuple): A tuple containing (prompt, action_1).

        Returns:
            float: Reward value for the specified action.
        """
        if not use_table:
            realizations = []
            for i in range(self.args['ensemble_size']): 
                rewards = self.compute_trajectory_rewards_from_model(trajectories, i, use_standardizer=use_standardizer)
                realizations.append(rewards)

            all_preds = torch.cat(realizations, dim=1)
            return all_preds.mean(dim=1), all_preds

        else:
            # input is shape [N, tMax, 2]
            N = trajectories.shape[0]
            E = self.args['ensemble_size']
            rewards = torch.zeros(N, E).to(self.device)
            for t in range(trajectories.shape[1]):
                states = trajectories[:,t,0]
                actions = trajectories[:,t,1]
                ensemble_indices = torch.arange(E).view(E, 1).expand(E, N)  # shape [E, N]
                state_indices = states.view(1, N).expand(E, N)              # shape [E, N]
                action_indices = actions.view(1, N).expand(E, N)            # shape [E, N]
                if use_standardizer:
                    rewards += self.stand_reward_table[ensemble_indices, state_indices, action_indices, t].T # shape [N, E]
                else:
                    rewards += self.raw_reward_table[ensemble_indices, state_indices, action_indices].T # shape [N, E]
            return rewards.mean(dim=1), rewards

    def generate_state_actions_reward(self, state_actions, t_index, use_standardizer=True):

        realizations = []
        for i in range(self.args['ensemble_size']): 
            rewards = self.compute_state_action_rewards_from_model(state_actions, t_index, i, use_standardizer=use_standardizer)
            realizations.append(rewards)

        all_preds = torch.cat(realizations, dim=1)
        return all_preds.mean(dim=1), all_preds
    
    def compute_trajectory_rewards_from_model(self, trajectories, model_index, use_standardizer=True):
        # input is shape [N, tMax, 2]
        rewards = torch.zeros(trajectories.shape[0], 1).to(self.device)
        for t in range(trajectories.shape[1]):
            timestep = trajectories[:,t,:]
            rewards_timestep = self.compute_state_action_rewards_from_model(timestep, t, model_index, use_standardizer=use_standardizer)
            rewards = rewards + rewards_timestep
        return rewards

    def compute_state_action_rewards_from_model(self, state_actions, t_index, model_index, use_standardizer=True):
        # input is shape [N, 2]
        one_hot_inputs = torch.cat((F.one_hot(state_actions[:,0], num_classes=36), F.one_hot(state_actions[:,1], num_classes=4)), dim = 1).float()
        rewards = self.models[model_index](one_hot_inputs)
        if use_standardizer:
            standardized_rewards = self.reward_standardizer.standardize(
                state_actions, t_index, rewards, model_index
            )
        else:
            standardized_rewards = rewards
        return standardized_rewards


    def train(self, preference_tuples, full_retrain=True):

        if full_retrain:
            del self.models
            self.models = []
            for i in range(self.args['ensemble_size']):
                self.models.append(SimpleMLP(**self.args['model']).to(self.device))

        self._add_data(preference_tuples)

        for i in range(self.args['ensemble_size']): 

            full_dataset = torch.utils.data.TensorDataset(self.full_data[0], self.full_data[1], self.full_data[2])
            if self.args['early_stopping']:
                train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [1-self.args['train_test_split'], self.args['train_test_split']])
                valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args['eval_batch_size'], shuffle=True)
            else:
                train_dataset = full_dataset
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args['train_batch_size'], shuffle=True)

            self.models[i].train()
            
            num_epochs = self.args['num_epochs']
            optimizer = optim.AdamW(self.models[i].parameters(), lr=self.args['lr'])
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

            # In-memory checkpoint (stores the best model state)
            best_model_state = None
            best_valid_loss = float('inf')  # Initialize best validation loss
            patience = self.args['early_stopping_patience']  # Number of epochs to wait for improvement
            early_stop_counter = 0
            
            for epoch in range(num_epochs):
                losses = []
                for batch_data in train_loader:
                    optimizer.zero_grad()
                    loss = self._compute_loss(i, batch_data)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach().cpu().numpy())
                scheduler.step()
                lr = optimizer.param_groups[0]['lr']

                if self.args['early_stopping']:
                    # Validation step
                    valid_losses = []
                    with torch.no_grad():
                        for batch_data in valid_loader:
                            valid_loss = self._compute_loss(i, batch_data)
                            valid_losses.append(valid_loss.detach().cpu().numpy())

                    avg_valid_loss = np.mean(valid_losses)
                    if epoch % 50 == 0:
                        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses):.4f}, "
                            f"Valid Loss: {avg_valid_loss:.4f}, LR: {lr:.6f}")

                    # Early stopping check
                    if avg_valid_loss < best_valid_loss:
                        best_valid_loss = avg_valid_loss
                        early_stop_counter = 0
                        # Save the best model state in memory (use deepcopy)
                        best_model_state = {
                            'model_state': deepcopy(self.models[i].state_dict()),
                            'optimizer_state': deepcopy(optimizer.state_dict())
                        }
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            print(f"Early stopping triggered. Best validation loss: {best_valid_loss}")

                            # Restore the best model state from memory
                            self.models[i].load_state_dict(best_model_state['model_state'])
                            optimizer.load_state_dict(best_model_state['optimizer_state'])

                            break
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses):.4f}, LR: {lr:.6f}")


    
    
    def _add_data(self, preference_tuples):
        #rows = torch.stack(preference_tuples, dim=1)
        if self.full_data is None:
            self.full_data = preference_tuples
        else: 
            self.full_data = (
                torch.cat((self.full_data[0], preference_tuples[0]), dim=0),
                torch.cat((self.full_data[1], preference_tuples[1]), dim=0),
                torch.cat((self.full_data[2], preference_tuples[2]), dim=0)
            )

    def _create_dataloader(self, train_data, batch_size):
        full_dataset = torch.utils.data.TensorDataset(train_data)
        return torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    def _select_chosen_rejected(self, inputs):
        trajectory1, trajectory2, labels = inputs[0], inputs[1], inputs[2]

        condition = labels.bool().reshape(trajectory1.shape[0], 1, 1).repeat(1, trajectory1.shape[1], trajectory1.shape[2])

        chosen_trajectory = torch.where(condition, trajectory1, trajectory2)
        rejected_trajectory = torch.where(condition, trajectory2, trajectory1)

        return chosen_trajectory, rejected_trajectory





    def _compute_loss(
            self,
            model_index,
            inputs):
        
        chosen, rejected = self._select_chosen_rejected(inputs)

        rewards_chosen = self.compute_trajectory_rewards_from_model(chosen, model_index, use_standardizer=False)
        rewards_rejected = self.compute_trajectory_rewards_from_model(rejected, model_index, use_standardizer=False)


        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # if self.regularized_loss:
        #     curr_params = [p.data for p in model.parameters()]
        #     l2_dist = sum((p1 - p2).norm(2).item() for p1, p2 in zip(curr_params, self.base_parameters))
        #     reg = self.lambda_reg / self.state.max_steps # Consider the number of gradient steps in the regularizer

        #     loss = loss + reg * l2_dist

        return loss
