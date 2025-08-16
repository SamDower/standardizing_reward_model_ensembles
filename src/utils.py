import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os
import GPUtil
import ast
import re

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_gpu_usage(str):
    usage = get_gpu_usage(str)
    for gpu in usage:
        print(f"GPU {gpu['id']}: {gpu['name']} - Load: {gpu['load']}%, Memory Utilization: {gpu['memoryUtil']}%, Temperature: {gpu['temperature']}°C")

def get_gpu_usage(str):
    gpus = GPUtil.getGPUs()
    gpu_usage = []
    print(f"GPU Usage {str}")
    for gpu in gpus:
        gpu_info = {
            'id': gpu.id,
            'name': gpu.name,
            'load': gpu.load * 100,
            'memoryUtil': gpu.memoryUtil * 100,
            'temperature': gpu.temperature
        }
        gpu_usage.append(gpu_info)
    return gpu_usage

def tensors_to_csv(tensors, keys, filename='output'):
    """
    Converts three PyTorch 1-D tensors to a CSV file.

    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
        tensor3 (torch.Tensor): Third tensor.
        filename (str, optional): Output CSV filename. Defaults to 'output.csv'.
    """
    # Convert tensors to Numpy arrays
    tensors_np = [tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]

    to_csv(keys, tensors_np, filename)

def to_csv(keys, arrays, filename):
    # Create a dictionary from the keys and arrays
    data = {}
    for key, array in zip(keys, arrays):
        if isinstance(array, np.ndarray) and array.ndim == 2:
            for i in range(array.shape[1]):
                data[f"{key}_{i}"] = array[:, i]
        else:
            data[key] = array

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV file
    df.to_csv(f'{filename}.csv', index=False)


def plot_curves_with_seaborn_and_save(csv_file, plot_file):
    """
    Reads a CSV file, plots two curves using Seaborn, and saves the figure as a JPG file.

    Args:
        filename (str): Path to the CSV file.
        output_filename (str, optional): Output filename for the saved figure. Defaults to 'curves_plot.jpg'.
    """
    # Read the CSV file
    df = pd.read_csv(f'{csv_file}.csv')

    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    # Plot y-values on the left axis
    ax1.plot(df.iloc[:, 0], df.iloc[:, 2], label='Gold', color='blue')
    ax1.set_xlabel('KL Budget')
    ax1.set_ylabel('Gold Reward')

    # Create a second y-axis for z-values
    
    ax2.plot(df.iloc[:, 0], df.iloc[:, 1], label='Proxy', color='red')
    ax2.set_ylabel('Proxy Reward')

    # Add title and legend
    plt.title('Reward Overoptimization')
    plt.legend()

    # Save the plot as a JPG file
    plt.savefig(plot_file, dpi=300)  # Adjust dpi as needed
    plt.close()

def plot_reward_scatter_with_seaborn_and_save(csv_file, plot_file):
    # Read the CSV file
    df = pd.read_csv(f'{csv_file}.csv')

    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Plot y-values on the left axis
    x_axis = df['Action'] + df['Prompt']
    rws_gold = df['RewardsGroundTruth']
    rws_proxy = df.filter(regex="^RewardsProxy").mean(axis=1)
    scaled_proxy_rws = (rws_proxy - rws_proxy.min()) / (rws_proxy.max() - rws_proxy.min())
    ax1.scatter(x_axis, scaled_proxy_rws, label='Proxy', color='red')
    ax1.scatter(x_axis, rws_gold, label='Gold', color='blue')
    ax1.set_xlabel('Context (prompt, action)')
    ax1.set_ylabel('Reward')

    # Add title and legend
    plt.title('Reward Latent')
    plt.legend()

    # Save the plot as a JPG file
    plt.savefig(plot_file, dpi=300)  # Adjust dpi as needed
    plt.close()

def plot_reward_scatter_with_seaborn_and_save(csv_file, plot_file):
    # Read the CSV file
    df = pd.read_csv(f'{csv_file}.csv')

    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Plot y-values on the left axis
    x_axis = df['Action'] + df['Prompt']
    rws_gold = df['RewardsGroundTruth']
    rws_proxy = df.filter(regex="^RewardsProxy").mean(axis=1)
    scaled_proxy_rws = (rws_proxy - rws_proxy.min()) / (rws_proxy.max() - rws_proxy.min())
    ax1.scatter(x_axis, scaled_proxy_rws, label='Proxy', color='red')
    ax1.scatter(x_axis, rws_gold, label='Gold', color='blue')
    ax1.set_xlabel('Context (prompt, action)')
    ax1.set_ylabel('Reward')

    # Add title and legend
    plt.title('Reward Latent')
    plt.legend()

    # Save the plot as a JPG file
    plt.savefig(plot_file, dpi=300)  # Adjust dpi as needed
    plt.close()

def plot_reward_curve_with_seaborn_and_save(csv_file, plot_file, poly_degree=5):
    # Read the CSV file
    df = pd.read_csv(f'{csv_file}.csv')

    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Plot y-values on the left axis
    x_axis = df['Action'] + df['Prompt']
    rws_gold = df['RewardsGroundTruth']
    rws_proxy = df.filter(regex="^RewardsProxy")
    scaled_proxy_rws = (rws_proxy - rws_proxy.min()) / (rws_proxy.max() - rws_proxy.min())

    ax1.scatter(x_axis, rws_gold, label='Gold', color='blue')

    if len(scaled_proxy_rws.columns) == 1:
        x_fit = np.linspace(x_axis.min(), x_axis.max(), 500)
        poly_proxy = np.poly1d(np.polyfit(x_axis, scaled_proxy_rws["RewardsProxy_0"], poly_degree))
        ax1.plot(x_fit, poly_proxy(x_fit), color='red', label='Proxy')
    else:
        x_fit = np.linspace(x_axis.min(), x_axis.max(), 500)
        rw_curves = []
        for rw_proxy in scaled_proxy_rws.columns:   
            poly_proxy = np.poly1d(np.polyfit(x_axis, scaled_proxy_rws[rw_proxy], poly_degree))
            rw_curves.append(poly_proxy(x_fit))
            ax1.plot(x_fit, poly_proxy(x_fit), color='red')

        # Plot fitted curves
        rw_curves_np = np.array(rw_curves)
        avg_rw = rw_curves_np.mean(axis=0)
        std_rw = rw_curves_np.std(axis=0)
        ax1.plot(x_fit, avg_rw, color='purple', label='Fitted Proxy')
        ax1.fill_between(x_fit, avg_rw - 2 * std_rw, avg_rw + 2 * std_rw, color='purple', alpha=0.3)
    
    ax1.set_xlabel('Context (prompt, action)')
    ax1.set_ylabel('Reward')

    # Add title and legend
    plt.title('Reward Latent')
    plt.legend()

    # Save the plot as a JPG file
    plt.savefig(plot_file, dpi=300)  # Adjust dpi as needed
    plt.close()


def plot_rewards(triples, gt_model, learned_model, plot_id):
    full_support_x = torch.linspace(0, 1.0, 1000).to(triples[0].device)
    full_support_y = torch.linspace(0, 1.0, 1000).to(triples[0].device)
    
    ys = learned_model.generate_reward((full_support_x, full_support_y)).detach().cpu().numpy()
    full_support_proxy_np = (full_support_x + full_support_y).detach().cpu().numpy()

    min_vals = ys.min()
    max_vals = ys.max()
    scaled_ys = (ys - min_vals) / (max_vals - min_vals)
    

    full_support_gt = torch.linspace(0, 2.0, 1000).to(triples[0].device)
    z = gt_model.generate_reward((full_support_gt, torch.zeros(full_support_x.shape).to(triples[0].device))).detach().cpu().numpy()
    x_np = full_support_gt.detach().cpu().numpy()
    plot_scatter_rewards(full_support_proxy_np, scaled_ys, x_np, z, filename=plot_id)


def plot_scatter_rewards(x, y, x_gt, z, filename='reward_scatter.png'):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')
    # Create a scatter plot for (x, y)
    plt.scatter(x, y, label="Proxy", color="blue")
    
    # Create a line plot for (x, z)
    plt.plot(x_gt, z, label="Ground Truth", color="red")
    
    # Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y and z")
    plt.legend()
    
    plt.savefig(filename, dpi=300)  # Adjust dpi as needed
    plt.close()

def plot_rate_over_data(data, metric, label, filename='metric.png'):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    plt.plot(data, metric, label=label, color="blue")

    # Add labels and legend
    plt.xlabel("Acquired Data")
    plt.ylabel(label)
    plt.legend()
    
    plt.savefig(filename, dpi=300)  # Adjust dpi as needed
    plt.close()

def plot_dataset_histograms(acquired_data, hist_prompts, bins_prompts, hist_acts, bins_acts, filepath_prefix):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    idx = 0
    for hist_prompt, bins_prompt, hist_acts, bins_acts in zip(hist_prompts, bins_prompts, hist_acts, bins_acts):
        values_prompt = []
        for count, bin_start, bin_end in zip(hist_prompt, bins_prompt[:-1], bins_prompt[1:]):
            values_prompt.extend([bin_start] * count)

        values_acts = []
        for count, bin_start, bin_end in zip(hist_acts, bins_acts[:-1], bins_acts[1:]):
            values_acts.extend([bin_start] * count)

        # Create the subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot the first histogram
        sns.histplot(values_prompt, bins=bins_prompt, kde=False, ax=axes[0])
        axes[0].set_title('States Distribution')
        axes[0].set_xlabel('States Value')
        axes[0].set_ylabel('Frequency')

        sns.heatmap(hist_prompt.reshape(6,6), ax=axes[1])
        axes[1].set_title('States Distribution')

        # Plot the second histogram
        sns.histplot(values_acts, bins=bins_acts, kde=False, ax=axes[2])
        axes[2].set_title('Actions Distribution')
        axes[2].set_xlabel('Action Value')
        axes[2].set_ylabel('Frequency')

        plt.savefig(f"{filepath_prefix}_{idx}", dpi=300)  # Adjust dpi as needed
        plt.close()
        idx = idx + 1

def plot_reward_tables(raw_reward_table, raw_reward_variances, standardized_reward_table, standardized_reward_variances, epoch, filepath_prefix):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    # Create the subplots
    fig, axes = plt.subplots(4, 4, figsize=(24, 24))

    for a in range(4):
        table = raw_reward_table[:,a].reshape(6,6)
        sns.heatmap(table, vmin=torch.min(raw_reward_table), vmax=torch.max(raw_reward_table), ax=axes[0,a])
        axes[0,a].set_title(f'R(s,{a})')
    
    for a in range(4):
        table = raw_reward_variances[:,a].reshape(6,6)
        sns.heatmap(table, vmin=torch.min(raw_reward_variances), vmax=torch.max(raw_reward_variances), ax=axes[1,a])
        axes[1,a].set_title(f'var(R(s,{a}))')
    
    for a in range(4):
        table = standardized_reward_table[:,a,0].reshape(6,6)
        sns.heatmap(table, vmin=torch.min(standardized_reward_table), vmax=torch.max(standardized_reward_table), ax=axes[2,a])
        axes[2,a].set_title(f'R_starc(s,{a})')
    
    for a in range(4):
        table = standardized_reward_variances[:,a,0].reshape(6,6)
        sns.heatmap(table, vmin=torch.min(standardized_reward_variances), vmax=torch.max(standardized_reward_variances), ax=axes[3,a])
        axes[3,a].set_title(f'var(R_starc(s,{a}))')

    plt.savefig(f"{filepath_prefix}_{epoch}", dpi=300)  # Adjust dpi as needed
    plt.close()
