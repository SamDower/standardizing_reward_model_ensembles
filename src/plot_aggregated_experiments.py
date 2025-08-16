import os
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def merge_seeds(exp_name, filename, columns):
    # Get a list of all seed directories
    
    seed_dirs = [d for d in os.listdir(exp_name) if os.path.isdir(os.path.join(exp_name, d))]

    # Initialize dictionaries to store dataframes for each column
    dfs_by_column = {col: {} for col in columns}

    # Load data from each seed
    for seed_dir in seed_dirs:
        raw_dir = os.path.join(exp_name, seed_dir, 'raw')
        file_path = os.path.join(raw_dir, filename)

        if os.path.exists(file_path):
            seed_df = pd.read_csv(file_path, usecols=columns)
            for col in columns:
                dfs_by_column[col][seed_dir] = seed_df[col]
        else:
            print(f"File '{filename}' not found in '{raw_dir}'.")

    # Create separate dataframes for each column
    result_dfs = {}
    for col, seed_data in dfs_by_column.items():
        result_dfs[col] = pd.DataFrame(seed_data)

    return result_dfs

def compute_mean_ci(y_df, x_df, metric):
    # Calculate the average of y
    y_metric = y_df.median(axis=1) if metric == 'median' else y_df.mean(axis=1)

    # Calculate the standard error of y
    stderr = y_df.apply(lambda x: stats.sem(x, axis=None, ddof=0), axis=1)

    # Set confidence level
    confidence = 0.95

    # Calculate the confidence interval for y
    ci = stderr * stats.t.ppf((1 + confidence) / 2., len(y_metric) - 1)

    x_axis = x_df.mean(axis=1)
    return x_axis, y_metric, ci

def plot_with_confidence_interval(metric, x_df, y_df, x_label, y_label, title):
    # Set Seaborn style (dark background)
    sns.set_style('darkgrid')
    # Set the color palette to 'colorblind'
    sns.set_palette('colorblind')

    x_axis, y_metric, ci = compute_mean_ci(y_df, x_df, metric)

    plt.plot(x_axis, y_metric, label=metric)
    plt.fill_between(x_axis, y_metric - ci, y_metric + ci, alpha=0.3)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()


def plot_with_confidence_interval_save(metric, x_df, y_df, x_label, y_label, title, filename):
    plot_with_confidence_interval(metric, x_df, y_df, x_label, y_label, title)
    plt.savefig(filename, dpi=300)
    plt.close()

def generating_x_axis(num_columns, batch_size, epochs):
    """
    Generates a DataFrame with 'num_columns' columns, each containing computed 'acquired_data'.

    Args:
        num_columns (int): Number of columns in the DataFrame.
        batch_size (int): Batch size for computation.
        epochs (int): Number of epochs for computation.

    Returns:
        pd.DataFrame: DataFrame with computed 'acquired_data' in each column.
    """
    acquired_data = np.arange(1, epochs + 1) * batch_size

    data_dict = {}
    for i in range(1, num_columns + 1):
        column_name = f"Column{i}"
        data_dict[column_name] = acquired_data

    return pd.DataFrame(data_dict)


def plot_files(exp_name, filename, columns, x_axis_col, metric='mean'):
    merged_data = merge_seeds(exp_name, filename, columns)
    x_axis = merge_seeds(exp_name, filename, [x_axis_col])[x_axis_col]
    
    for col, df in merged_data.items():
        print(f"Dataframe for column '{col}':\n{df}\n")
        filepath = os.path.join(agg_plots_dir, f"{col} - {metric}")
        plot_with_confidence_interval_save(metric, x_axis, df, x_axis_col, col, col, filepath)

def plot_experiment(exp_name_prefix, exp_names, exp_labels, agg_plots_dir, filename, columns, x_axis_col, metric='mean'):
    exps = []
    for exp in exp_names:
        exp_dir = os.path.join(exp_name_prefix, exp)
        merged_data = merge_seeds(exp_dir, filename, columns)
        x_axis = merge_seeds(exp_dir, filename, [x_axis_col])[x_axis_col]
        exps.append(merged_data)

    for col in columns:
        # Set Seaborn style (dark background)
        sns.set_style('darkgrid')
        # Set the color palette to 'colorblind'
        sns.set_palette('colorblind')
        filepath = os.path.join(agg_plots_dir, f"{col} - {metric}")
        for exp, exp_label in zip(exps, exp_labels):
            print(f"Plotting'{col} for {exp_label}")
            
            x, y_metric, ci = compute_mean_ci(exp[col], x_axis, metric)

            plt.plot(x, y_metric, label=exp_label)
            plt.fill_between(x, y_metric - ci, y_metric + ci, alpha=0.3)

        plt.xlabel(x_axis_col)
        plt.ylabel(col)
        plt.title(col)
        plt.legend()
        plt.savefig(filepath, dpi=300)
        plt.close()

def plot_accuracies(title, exp_name_prefix, exp_names, exp_labels, agg_plots_dir, filename, columns, x_axis_col, metric='mean'):
    exps = []
    for exp in exp_names:
        exp_dir = os.path.join(exp_name_prefix, exp)
        merged_data = merge_seeds(exp_dir, filename, columns)
        x_axis = merge_seeds(exp_dir, filename, [x_axis_col])[x_axis_col]
        exps.append(merged_data)

    for col in columns:
        # Set Seaborn style (dark background)
        sns.set_style('darkgrid')
        # Set the color palette to 'colorblind'
        sns.set_palette('colorblind')
        filepath = os.path.join(agg_plots_dir, f"{col} - {metric}")
        for exp, exp_label in zip(exps, exp_labels):
            print(f"Plotting'{col} for {exp_label}")
            
            x, y_metric, ci = compute_mean_ci(exp[col], x_axis, metric)

            plt.plot(x, y_metric, label=exp_label)
            plt.fill_between(x, y_metric - ci, y_metric + ci, alpha=0.3)

        plt.xlabel(x_axis_col)
        plt.ylabel("Preference Accuracy")
        plt.title(title)
        plt.legend()
        plt.savefig(filepath, dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge data from different seeds.")
    
        # Naive
    # title = "Test Accuracy - Naive Approach"
    # exp_names = ['36_15_varp_none', '36_15_varp_starc', '36_15_random']#, '36_15_sumvarprel_starc']#,  '36_15_sumvarprel_starc', '36_15_sumvarr_starc']#, 'varp_random_starc_scale100', 'varp_random_starc_scale500', 'varp_random_starc_scale800']#, 'random_random']#, 'bald_exp_starcnonorm_bi_mc']
    # exp_labels = ['EnsembleVar', 'EnsembleVar + Standardizing', 'Random']#, 'sum_starc']#, 'sumvarprel_starc', 'sumvarr_starc']#, 'baseline']#, 'nonorm']

        # Beta hyperparameter search
    title = "Test Accuracy - $\\beta$ hyperparameter search"
    exp_names = ['36_15_varp_starc', 'mag_hyp_starc_500', 'mag_hyp_starc_5000', 'mag_hyp_starc_50000', 'mag_hyp_starc_500000']
    exp_labels = ['$\\beta=1$', '$\\beta=500$', '$\\beta=5000$', '$\\beta=50000$', '$\\beta=500000$']#, 'baseline']#, 'nonorm']

    title = "Test"
    exp_names = ['36_15_varprel_starc', '36_15_varp_starc', 'mag_hyp_starc_500', 'mag_hyp_starc_5000', 'mag_hyp_starc_50000', 'mag_hyp_starc_500000']
    exp_labels = ['rel', '$\\beta=1$', '$\\beta=500$', '$\\beta=5000$', '$\\beta=50000$', '$\\beta=500000$']#, 'baseline']#, 'nonorm']


        # Rel
    # title = "Test Accuracy - Relative Rescaling"
    # exp_names = ['36_15_varp_none', '36_15_varprel_starc', '36_15_random']#, '36_15_sumvarprel_starc']#,  '36_15_sumvarprel_starc', '36_15_sumvarr_starc']#, 'varp_random_starc_scale100', 'varp_random_starc_scale500', 'varp_random_starc_scale800']#, 'random_random']#, 'bald_exp_starcnonorm_bi_mc']
    # exp_labels = ['EnsembleVar', 'EnsembleVar + Standardizing', 'Random']#, 'sum_starc']#, 'sumvarprel_starc', 'sumvarr_starc']#, 'baseline']#, 'nonorm']

        # Aleatoric
    title = "Test Accuracy - Adding Aleatoric Uncertainty"
    #exp_names = ['36_15_varp_none', '36_15_varprel_starc', '36_15_al10_varp_none', '36_15_al10_varprel_starc', '36_15_al20_varp_none', '36_15_al20_varprel_starc']
    exp_names = ['36_15_varp_none', '36_15_varprel_starc', '36_15_al10_varp_none_oldset', '36_15_al10_varprel_starc_oldset', '36_15_al20_varp_none_oldset', '36_15_al20_varprel_starc_oldset']
    exp_labels = ['noise=0%    EnsembleVar', 'noise=0%    EnsembleVar + Standardizing', 'noise=10%  EnsembleVar', 'noise=10%  EnsembleVar + Standardizing', 'noise=20%  EnsembleVar', 'noise=20%  EnsembleVar + Standardizing']


        # Non Deterministic
    # title = "Test Accuracy - Non-Deterministic Environment"
    # exp_names = ['36nd_15_varp_none', '36nd_15_varprel_starc', '36nd_15_al10_varp_none', '36nd_15_al10_varprel_starc', '36nd_15_al20_varp_none', '36nd_15_al20_varprel_starc']#, '36_15_sumvarprel_starc']#,  '36_15_sumvarprel_starc', '36_15_sumvarr_starc']#, 'varp_random_starc_scale100', 'varp_random_starc_scale500', 'varp_random_starc_scale800']#, 'random_random']#, 'bald_exp_starcnonorm_bi_mc']
    # exp_labels = ['none', 'starc', '10_none', '10_starc', '20_none', '20_starc']#, 'sum_starc']#, 'sumvarprel_starc', 'sumvarr_starc']#, 'baseline']#, 'nonorm']

    # title = "Test Accuracy - Non-Deterministic Environment"
    # exp_names = ['36nd_15_varp_none', '36nd_15_varprel_starc', '36nd_15_random']
    # exp_labels = ['EnsembleVar', 'EnsembleVar + Standardizing', 'Random']


    #     # Sum Varr
    # title = "Test Accuracy - TransitionSum Comparison"
    # exp_names = ['36_15_sumvarr_none', '36_15_sumvarr_starc']
    # exp_labels = ['TransitionSum', 'TransitionSum + Standardizing']


    # exp_names = ['36_15_varp_none_oldset', '36_15_varprel_starc_oldset', '36_15_al10_varp_none_oldset', '36_15_al10_varprel_starc_oldset', '36_15_al20_varp_none_oldset', '36_15_al20_varprel_starc_oldset']#, '36_15_sumvarprel_starc']#,  '36_15_sumvarprel_starc', '36_15_sumvarr_starc']#, 'varp_random_starc_scale100', 'varp_random_starc_scale500', 'varp_random_starc_scale800']#, 'random_random']#, 'bald_exp_starcnonorm_bi_mc']
    # exp_labels = ['none', 'starc', '10_none', '10_starc', '20_none', '20_starc']#, 'sum_starc']#, 'sumvarprel_starc', 'sumvarr_starc']#, 'baseline']#, 'nonorm']
    # exp_names = ['rew36_15_varr_none', 'rew36_15_varr_starc', 'rew36_15_sumvarr_none', 'rew36_15_sumvarr_starc']
    # exp_labels = ['none', 'starc', 'sum-none', 'sum-starc']
    # exp_names = ['36_15_varp_none_oldset', '36_15_varprel_starc_oldset']#,  '36_15_sumvarprel_starc', '36_15_sumvarr_starc']#, 'varp_random_starc_scale100', 'varp_random_starc_scale500', 'varp_random_starc_scale800']#, 'random_random']#, 'bald_exp_starcnonorm_bi_mc']
    # exp_labels = ['varp_none', 'varprel_starc']#, 'sumvarprel_starc', 'sumvarr_starc']#, 'baseline']#, 'nonorm']
    

    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    exp_name_prefix = os.path.join(script_dir, "../experiments")
    agg_plots_dir = filepath = os.path.join(script_dir, "../experiments", "aggregated_plots")
    os.makedirs(agg_plots_dir, exist_ok=True)
    plot_experiment(exp_name_prefix, exp_names, exp_labels, agg_plots_dir, "loglikelihoods.csv", ["Log Likelihood (Train)", "Log Likelihood (Test)"], "Acquired Data")
    #plot_experiment(exp_name_prefix, exp_names, exp_labels, agg_plots_dir, "accuracies.csv", ["Accuracy (Train)", "Accuracy (Test Raw)", "Accuracy (Test Stand)"], "Acquired Data")
    #plot_experiment(exp_name_prefix, exp_names, exp_labels, agg_plots_dir, "win_rates.csv", ["Win Rate (Train)", "Win Rate (Test)"], "Acquired Data")
    #plot_experiment(exp_name_prefix, exp_names, exp_labels, agg_plots_dir, "win_rates.csv", ["Win Rate (Train)", "Win Rate (Test)"], "Acquired Data", metric='median')
    #plot_experiment(exp_name_prefix, exp_names, exp_labels, agg_plots_dir, "cvar_win_rates.csv", ["Win Rate (CVaR - Train)", "Win Rate (CVaR - Test)"], "Acquired Data")
    plot_accuracies(title, exp_name_prefix, exp_names, exp_labels, agg_plots_dir, "accuracies.csv", ["Accuracy (Train)", "Accuracy (Test Raw)", "Accuracy (Test Stand)"], "Acquired Data")
    
    


