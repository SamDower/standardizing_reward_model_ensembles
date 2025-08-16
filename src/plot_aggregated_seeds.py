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


def plot_with_confidence_interval(metric, x_df, y_df, x_label, y_label, title, filename):
    # Set Seaborn style (dark background)
    sns.set_theme(style='darkgrid')

    # Calculate the average of y
    y_metric = y_df.median(axis=1) if metric == 'median' else y_df.mean(axis=1)

    # Calculate the standard error of y
    stderr = y_df.apply(lambda x: stats.sem(x, axis=None, ddof=0), axis=1)

    # Set confidence level
    confidence = 0.95

    # Calculate the confidence interval for y
    ci = stderr * stats.t.ppf((1 + confidence) / 2., len(y_metric) - 1)

    x_axis = x_df.mean(axis=1)
    plt.plot(x_axis, y_metric, label=metric)
    plt.fill_between(x_axis, y_metric - ci, y_metric + ci, alpha=0.3)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
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
        plot_with_confidence_interval(metric, x_axis, df, x_axis_col, col, col, filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge data from different seeds.")
    parser.add_argument("--exp_name", type=str, help="Experiment name (directory containing seed folders)")
    parser.add_argument("--filename", type=str, help="CSV filename to load from each seed's raw folder")
    parser.add_argument("--columns", nargs="+", help="Columns to read from the CSV file")

    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    exp_name = os.path.join(script_dir, "../experiments", args.exp_name)
    agg_plots_dir = os.path.join(exp_name, "aggregated_plots")
    os.makedirs(agg_plots_dir, exist_ok=True)
    plot_files(exp_name, "loglikelihoods.csv", ["Log Likelihood (Train)", "Log Likelihood (Test)"], "Acquired Data")
    plot_files(exp_name, "accuracies.csv", ["Accuracy (Train)", "Accuracy (Test)"], "Acquired Data")
    plot_files(exp_name, "win_rates.csv", ["Win Rate (Train)", "Win Rate (Test)"], "Acquired Data")
    plot_files(exp_name, "win_rates.csv", ["Win Rate (Train)", "Win Rate (Test)"], "Acquired Data", metric='median')
    plot_files(exp_name, "cvar_win_rates.csv", ["Win Rate (CVaR - Train)", "Win Rate (CVaR - Test)"], "Acquired Data")

    


