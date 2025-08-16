from sklearn.model_selection import train_test_split
from modules import MCDropoutRewardMLP
from preference_models import UncertaintyAwarePreferenceModel
from reward_standardizers import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

class MCDropoutPreferenceModel(UncertaintyAwarePreferenceModel):
    def __init__(self, device, reward_standardizer, **kwargs):
        self.args = kwargs
        self.device = device
        self.reward_standardizer = reward_standardizer
        self.model = MCDropoutRewardMLP(**self.args['model']).to(device)
        self.standardize = kwargs['standardize_features']
        self.full_data = None
        self._data_min = None
        self._data_max = None

    def generate_preferences(self, prompt_response_triple, policy_for_starc=None, override_standardizer=False):
        return super().generate_preferences(prompt_response_triple, policy_for_starc=policy_for_starc, override_standardizer=override_standardizer)


    def generate_reward(self, prompt_action_pair, features=True, policy_for_starc=None, override_standardizer=False):
        """
        Generates a reward based on the given prompt-action pair.

        Args:
            prompt_action_pair (tuple): A tuple containing (prompt, action_1).

        Returns:
            float: Reward value for the specified action.
        """
        inputs = self._generate_reward_model_inputs(prompt_action_pair, features)
        if self.standardize:
            if self._data_min is not None:
                inputs, _, _ = self._standardize_features(inputs, self._data_min, self._data_max)
       
        realizations = []
        for _ in range(self.args['mc_dropout_realizations']): # MC Dropout Realizations
            if True:
                self.model.freeze()
                rewards = self.model(inputs)
                if not override_standardizer:
                    standardized_rewards = self.reward_standardizer.standardize(
                        inputs, rewards, self.model, policy_for_starc
                    )
                else:
                    standardized_rewards = rewards
                self.model.unfreeze()
                realizations.append(standardized_rewards)
            else: # Canonicalise the reward function
                self.model.freeze()
                with torch.no_grad():
                    # Sample actions from policy (context doesn't matter in simple setup)
                    rand_actions = policy_for_starc.act(torch.ones(100).to(self.device))
                    print(rand_actions.shape)
                    # For each prompt
                    expected_rewards = torch.tensor([]).to(self.device)
                    for i in range(len(prompt_action_pair[0])):
                        # Contruct a new prompt-action pairs
                        pa = (torch.ones(100).to(self.device)*prompt_action_pair[0][i], rand_actions)
                        pa_inputs = self._generate_reward_model_inputs(pa, features)
                        if self.standardize:
                            if self._data_min is not None:
                                pa_inputs, _, _ = self._standardize_features(pa_inputs, self._data_min, self._data_max)
                        # Get rewards for these sampled actions
                        pa_rewards = self.model(pa_inputs)
                        # Calculate expected reward for this prompt
                        expected_reward = pa_rewards.mean().reshape([1])
                        expected_rewards = torch.cat((expected_rewards, expected_reward))
                        del expected_reward, pa_rewards, pa_inputs, pa
                    expected_rewards = expected_rewards.reshape((len(prompt_action_pair[0]), 1))
                    rewards = self.model(inputs)
                    print(f"rewards: {rewards.shape}")
                    starc_rewards = rewards - expected_rewards
                    #print(f"starc rewards {starc_rewards.shape}")
                    #print(f"max: {starc_rewards.max()}")
                    #print(f"min: {starc_rewards.min()}")
                    starc_rewards = starc_rewards / (starc_rewards.max() - starc_rewards.min())
                    self.model.unfreeze()
                    realizations.append(starc_rewards)
                    del expected_rewards, rewards, starc_rewards

        #gc.collect()
        #torch.cuda.empty_cache()
        all_preds = torch.cat(realizations, dim=1)
        return all_preds.mean(dim=1), all_preds

    def train(self, preference_tuples, full_retrain=True):
        if full_retrain:
            del self.model
            self.model = MCDropoutRewardMLP(**self.args['model']).to(self.device)

        preference_tuples = self._preprocess_data(preference_tuples)
        self._add_data(preference_tuples)
        if self.standardize:
            dataset, self._data_min, self._data_max = self._standardize_triplets(self.full_data) 
        else: 
            dataset = self.full_data   

        if self.args['early_stopping']:
            train, valid = train_test_split(dataset, test_size=self.args['train_test_split'])
            valid_loader = self._create_dataloader(valid, batch_size=self.args['eval_batch_size'])
        else:
            train = dataset

        train_loader = self._create_dataloader(train, batch_size=self.args['train_batch_size'])


        self.model.train()
        
        num_epochs = self.args['num_epochs']
        best_valid_loss = float('inf')  # Initialize best validation loss
        patience = self.args['early_stopping_patience']  # Number of epochs to wait for improvement
        early_stop_counter = 0
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        for epoch in range(num_epochs):
            losses = []
            for batch_data in train_loader:
                optimizer.zero_grad()
                loss = self._compute_loss(self.model, batch_data[0])
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
                        valid_loss = self._compute_loss(self.model, batch_data[0])
                        valid_losses.append(valid_loss.detach().cpu().numpy())

                avg_valid_loss = np.mean(valid_losses)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses):.4f}, "
                    f"Valid Loss: {avg_valid_loss:.4f}, LR: {lr:.6f}")

                # Early stopping check
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping triggered.")
                        break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses):.4f}, LR: {lr:.6f}")


    def _preprocess_data(self, pref_tuples):
        prompt_action_1, prompt_action_2, labels = pref_tuples
        prompt, action_1 = prompt_action_1
        _, action_2 = prompt_action_2
        return (prompt, action_1, action_2, labels)
    
    def _standardize_triplets(self, tensor):
        prompts, a1, a2, pref_labels = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]
        scaled_prompts, min_prompt, max_prompt = self._standardize_features(prompts)
        actions = torch.cat((a1, a2), dim=0)
        scaled_actions, min_act, max_act = self._standardize_features(actions)
        scaled_a1, scaled_a2 = torch.split(scaled_actions, split_size_or_sections=scaled_actions.size(0) // 2, dim=0)

        # Combine the scaled columns with the remaining column
        scaled_full_tensor = torch.cat([scaled_prompts.unsqueeze(1), scaled_a1.unsqueeze(1), scaled_a2.unsqueeze(1), pref_labels.unsqueeze(1)], dim=1)
        min_vals = torch.cat((min_prompt.reshape(1), min_act.reshape(1)))
        max_vals = torch.cat((max_prompt.reshape(1), max_act.reshape(1)))

        del scaled_prompts, min_prompt, max_prompt, actions, scaled_actions, min_act, max_act, scaled_a1, scaled_a2
        return scaled_full_tensor, min_vals, max_vals
    
    def _standardize_features(self, tensor, min=None, max=None):
        if min is None or max is None:
            min_vals, _ = tensor.min(dim=0)
            max_vals, _ = tensor.max(dim=0)
        else:
            min_vals = min
            max_vals = max
        scaled_tensor = (tensor - min_vals) / (max_vals - min_vals)
        return scaled_tensor, min_vals, max_vals
    
    def _add_data(self, preference_tuples):
        rows = torch.stack(preference_tuples, dim=1)
        if self.full_data is None:
            self.full_data = rows
        else: 
            self.full_data = torch.cat((self.full_data, rows), dim=0)

    def _create_dataloader(self, train_data, batch_size):
        full_dataset = torch.utils.data.TensorDataset(train_data)
        return torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    def _select_chosen_rejected(self, inputs):
        prompt, action1, action2, labels = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]

        chosen_action = torch.where(labels.bool(), action1, action2)
        rejected_action = torch.where(labels.bool(), action2, action1)

        chosen = torch.cat((prompt.unsqueeze(1), chosen_action.unsqueeze(1)), dim=1)
        rejected = torch.cat((prompt.unsqueeze(1), rejected_action.unsqueeze(1)), dim=1)

        del chosen_action, rejected_action
        return chosen, rejected
    
    def _generate_reward_model_inputs(self, prompt_action_pair, features=True):
        prompt, action = prompt_action_pair
        inputs = torch.cat((prompt.unsqueeze(1), action.unsqueeze(1)), dim=1)
        return inputs

    def _compute_loss(
            self,
            model,
            inputs):
        
        chosen, rejected = self._select_chosen_rejected(inputs)
    
        rewards_chosen = model(
            chosen
        )
        rewards_rejected = model(
            rejected
        )

        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # if self.regularized_loss:
        #     curr_params = [p.data for p in model.parameters()]
        #     l2_dist = sum((p1 - p2).norm(2).item() for p1, p2 in zip(curr_params, self.base_parameters))
        #     reg = self.lambda_reg / self.state.max_steps # Consider the number of gradient steps in the regularizer

        #     loss = loss + reg * l2_dist

        return loss
        
        

