from trajectory_datasets import *
from markov_desicion_processes import *
from data_selection_policies import *
from policies import *
import torch
from configs import * 
from preference_models import *
from reward_standardizers import *
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

def generateTraj(policy, mdp):
    currentState = mdp.get_initial_state()
    states = [currentState]
    reward = 0
    for t in range(mdp.tMax):
        action = torch.argmax(policy.Q_table[currentState,:])
        nextState = mdp.get_next_state(torch.tensor([currentState, action]))
        reward += mdp.get_reward(torch.tensor([currentState, action]))
        states.append(nextState.item())
        currentState = nextState
    return states, reward


def generateTraj2(values, mdp, device):
    currentState = mdp.get_initial_state()
    states = [currentState]
    reward = 0
    for t in range(mdp.tMax):
        action = torch.argmax(torch.tensor([
            mdp.get_reward(torch.tensor([currentState, 0]).to(device)) + values[mdp.get_next_state(torch.tensor([currentState, 0]).to(device))],
            mdp.get_reward(torch.tensor([currentState, 1]).to(device)) + values[mdp.get_next_state(torch.tensor([currentState, 1]).to(device))],
            mdp.get_reward(torch.tensor([currentState, 2]).to(device)) + values[mdp.get_next_state(torch.tensor([currentState, 2]).to(device))],
            mdp.get_reward(torch.tensor([currentState, 3]).to(device)) + values[mdp.get_next_state(torch.tensor([currentState, 3]).to(device))],
        ]))#policy.Q_table[currentState,:])
        nextState = mdp.get_next_state(torch.tensor([currentState, action]))
        reward += mdp.get_reward(torch.tensor([currentState, action]))
        states.append(nextState.item())
        currentState = nextState
    return states, reward

def sigmoid(x):
    return 1/(1+np.exp(-x))

def points_around(x):
    return np.array([x-0.5,x-0.25,x,x+0.25,x+0.5])

if __name__ == "__main__":

    # # Set Seaborn style (dark background)
    # sns.set_theme(style='darkgrid')

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Plot y-values on the left axis
    x = np.arange(-5,5,step=0.1)
    ax1.plot(x, sigmoid(x), c="black", linewidth=4.0)

    d = 2.5

    ax1.scatter(points_around(0), np.zeros(5), c="blue")
    ax1.scatter(points_around(-d), np.zeros(5), c="red")
    ax1.scatter(points_around(d), np.zeros(5), c="green")

    ax1.scatter(np.ones(5)*-5, sigmoid(points_around(0)), facecolors='none', edgecolors='blue')
    ax1.scatter(np.ones(5)*-5, sigmoid(points_around(-d)), facecolors='none', edgecolors='red')
    ax1.scatter(np.ones(5)*-5, sigmoid(points_around(d)), facecolors='none', edgecolors='green')

    ax1.vlines(points_around(0), 0, sigmoid(points_around(0)), colors='blue', linestyles='dashed', alpha=0.5)
    ax1.vlines(points_around(-d), 0, sigmoid(points_around(-d)), colors='red', linestyles='dashed', alpha=0.5)
    ax1.vlines(points_around(d), 0, sigmoid(points_around(d)), colors='green', linestyles='dashed', alpha=0.5)

    ax1.hlines(sigmoid(points_around(0)), -5, points_around(0), colors='blue', linestyles='dashed', alpha=0.5)
    ax1.hlines(sigmoid(points_around(-d)), -5, points_around(-d), colors='red', linestyles='dashed', alpha=0.5)
    ax1.hlines(sigmoid(points_around(d)), -5, points_around(d), colors='green', linestyles='dashed', alpha=0.5)

    # ax1.set_xlabel('KL Budget')
    # ax1.set_ylabel('Gold Reward')

    # Create a second y-axis for z-values

    # Add title and legend
    plt.title('Sigmoid Function')

    # Save the plot as a JPG file
    plt.savefig("sigmoid", dpi=300)  # Adjust dpi as needed
    plt.close()


    # base_config = load_config("base_config")
    # config = load_config("36_15_varp_none")
    # config = override_config(base_config, config)

    # device = torch.device('cuda')
    # markov_decision_process = LavaPathDetMDP(config, device)
    # # reward_standardizer = StarcRewardStandardizer(config['reward_standardizer'], device, markov_decision_process)
    # # pref_model = EnsemblePreferenceModel(device, reward_standardizer, **config['pref_model'])

    # ax = sns.heatmap(markov_decision_process.rewards.detach().cpu().reshape(6,6), annot=True, linewidth=.5, fmt='g', annot_kws={'size': 15})#, yticklabel=False, xticklabel=False)
    # ax.tick_params(left=False, bottom=False)
    # plt.savefig("grid", dpi=300) 


    # for j in range(40, 50):
    #     models_path = f"experiments/36_15_varp_none/5270/raw/reward_model_epoch_{j}_"
    #     for i in range(10):
    #         pref_model.models[i].load_state_dict(torch.load(models_path + f"{i}", weights_only=True))
    #         pref_model.models[i].eval()
        
    #     pref_model.generate_stand_reward_table()
    #     print(pref_model.stand_reward_table.mean(dim=0).shape)
    #     print(torch.sum(torch.abs(pref_model.stand_reward_table.mean(dim=0))))

    # dataset = RandomNonDetTrajectoryDataset(None, markov_decision_process, device)
    # dataset.generate_dataset(50)
    # print(dataset.samples[0][0])








    # Create the subplots
    # sns.set_theme(style='darkgrid')
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # device = torch.device('cuda')
    # markov_decision_process = LavaPathMDP(device)

    # sns.heatmap(markov_decision_process.rewards.reshape(6,6).cpu(), ax=axes[1] ,annot=True, fmt='g', annot_kws={'size': 10})
    # axes[1].set_title('States Distribution')


    # plt.savefig(f"test_image", dpi=300)  # Adjust dpi as needed
    # plt.close()

    # x = np.array([0, 1, 1, 0, 0, 4])
    # print(np.histogram(x, bins=8, range=(0,8)))

    # y = torch.ones(5, 16, 2)
    # print(y.shape)
    # print(y[:,0,:].shape)

    # x = torch.randint(5, (5,2))
    # print(x)

    # a = F.one_hot(x[:,0], num_classes=7)
    # b = F.one_hot(x[:,1], num_classes=5)
    # print(torch.cat((a, b), dim = 1))
    # print(torch.cat((a, b), dim = 1).shape)

    # cond = torch.tensor([1, 0, 0, 1, 0]).bool()
    # print(cond)
    # print(cond.shape)
    # x = cond.reshape(5, 1, 1)
    # print(x)
    # print(x.shape)
    # x = x.repeat(1,4,2)
    # print(x)
    # print(x.shape)


    #probs[dist.sample()]

    # x = torch.tensor([0, 0, 0])
    # probs = torch.tensor([0.03, 0.13, 0.84], requires_grad=True)
    # dist=torch.distributions.categorical.Categorical(probs=probs)
    # for i in range(100000):
    #     #idx = torch.searchsorted(p, torch.rand(1))
    #     x[dist.sample()] += 1
    # print(x / 100000)

    # for i in range(10):
    #     print(torch.rand(1))

    # device = torch.device('cuda')

    # markov_decision_process = LavaPathMDP(device)
    # policy = BoltzmannPolicy(None, markov_decision_process, device)
    # policy.train_sarsa_on_mdp()
    # # value_iteration = ValueIteration(markov_decision_process, device)
    # # value_iteration.perform_value_iteration_on_mdp()
    # # print(value_iteration.values.reshape(6,6))

    # print(policy.act_out_trajectories(1)[:,:,0])
    # print(policy.act_boltzmann_out_trajectories(5)[:,:,0])

    # for i in range(5):
    #     traj, rew = generateTraj(policy, markov_decision_process)
    #     #traj, rew = generateTraj2(value_iteration.values, markov_decision_process, device)
    #     print(traj)
    #     print(rew)




















    # device = torch.device('cuda')
    # config = load_config("base_config")

    # seed = config['seed']

    # markov_decision_process = LavaPathDetMDP(device)
    # dataset = RandomTrajectoryDataset(None, markov_decision_process, device)
    # dataset.generate_dataset(1000)

    # standardizer = GTStarcRewardStandardizer(device, markov_decision_process)
    # standardizer.calculate_values()
    # standardizer.calculate_norms()

    # gt_rewards = torch.zeros(36, 4)
    # for s in range(36):
    #     for a in range(4):
    #         gt_rewards[s][a] = markov_decision_process.get_reward(torch.tensor([s,a]).to(device))
    # standardised_rewards = torch.zeros(36, 4)
    # for s in range(36):
    #     for a in range(4):
    #         standardised_rewards[s][a] = standardizer.standardize(torch.tensor([s,a]).reshape(1,2).to(device)).item()
    

    # print(gt_rewards[:,0].reshape(6,6))
    # print(standardised_rewards[:,0].reshape(6,6))
    # #print(standardised_rewards[:,0].reshape(6,6))
    # print(torch.mean(standardised_rewards, dim=1).reshape(6,6))
    


    # print(torch.max(gt_rewards, dim=1).values.reshape(6,6) - torch.min(gt_rewards, dim=1).values.reshape(6,6))
    # #print(standardised_rewards[:,0].reshape(6,6))
    # print(torch.max(standardised_rewards, dim=1).values.reshape(6,6) - torch.min(standardised_rewards, dim=1).values.reshape(6,6))
    


















    # sum = 0
    # for i in range(len(dataset.samples[0])):
    #     gt_rews = []
    #     rews = []
    #     for k in range(2):
    #         trajectory = dataset.samples[k][i]
    #         gt_trajectory_reward = 0
    #         trajectory_reward = 0
    #         for t in range(len(trajectory)):
    #             sa = trajectory[t].reshape(1,3)[:,:2]
    #             gt_reward = markov_decision_process.get_reward(trajectory[t])
    #             reward = standardizer.standardize(sa, t)
    #             gt_trajectory_reward += gt_reward
    #             trajectory_reward += reward
    #         gt_rews.append(gt_trajectory_reward)
    #         rews.append(trajectory_reward.reshape(1))
        
    #     if (gt_rews[0] <= gt_rews[1]) == (rews[0][0] <= rews[1][0]):
    #         sum += 1
    #     else:
    #         print(f"GT: {gt_rews[0] - gt_rews[1]}")
    #         print(f"norm: {rews[0][0] - rews[1][0]}")
    
    # print(f"Agree_rate: {sum / len(dataset.samples[0])}")
            

    # value_iteration = ValueIteration(markov_decision_process, device)
    # value_iteration.perform_value_iteration(markov_decision_process.get_reward)
    # print(value_iteration.values.reshape(6,6))

    # data_selection_policy = RandomSelectionPolicy(None)

    # trajectory_pairs = data_selection_policy.select_data(dataset, None, batch_size=20)

    # preference_labels, _, _, _ = markov_decision_process.generate_gt_preferences(trajectory_pairs)
    # preference_tuples = trajectory_pairs + (preference_labels,)

    # preference_model = EnsemblePreferenceModel(device, None, **config['pref_model'])
    # preference_model.train(preference_tuples, full_retrain=True)







    # pairs = data_selection_policy.select_data(dataset, None, 3)

    # gt_preferences, _, first, second = mdp.generate_gt_preferences(pairs)
    # print(first)
    # print(second)
    # print(gt_preferences)



    # 2. Generate Preferences from Oracle
    #preference_labels, _, _, _ = self.gt_preference_model.generate_preferences(triples, mode="train")