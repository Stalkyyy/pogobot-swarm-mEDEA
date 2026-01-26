#!/usr/bin/env python3
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import glob
import os
import re
import sys
from scipy.spatial.distance import pdist
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_runs(runs_dir):
    run_folders = sorted([f for f in glob.glob(os.path.join(runs_dir, "run_*")) if os.path.isdir(f)])
    
    if not run_folders:
        print(f"No run folders found in {runs_dir}")
        return None, None

    all_pos_list = []
    all_agents_list = []

    print(f"Found {len(run_folders)} runs in {runs_dir}. Loading all data...")

    for run_path in run_folders:
        # Extract run ID from folder name (e.g., runs/run_1 -> 1)
        try:
            run_id = int(re.search(r'run_(\d+)', run_path).group(1))
        except:
            run_id = 0
            
        # Position Data
        feather_path = os.path.join(run_path, 'data.feather')
        if os.path.exists(feather_path):
            df = feather.read_feather(feather_path)
            df['run_id'] = run_id
            all_pos_list.append(df)

        # Agent Logs
        agent_files = glob.glob(os.path.join(run_path, 'agent_log_*.csv'))
        for file in agent_files:
            try:
                robot_id = int(file.split('_')[-1].split('.')[0])
                df = pd.read_csv(file)
                df['robot_id'] = robot_id
                df['run_id'] = run_id
                all_agents_list.append(df)
            except Exception as e:
                print(f"Skipping corrupt file {file}: {e}")

    df_pos_all = pd.concat(all_pos_list, ignore_index=True) if all_pos_list else None
    df_agents_all = pd.concat(all_agents_list, ignore_index=True) if all_agents_list else None

    if df_pos_all is not None:
        print(f"Total position records: {len(df_pos_all)}")
    if df_agents_all is not None:
        print(f"Total agent records: {len(df_agents_all)}")
        # Unique identifier string for plotting
        df_agents_all['uid'] = "R" + df_agents_all['run_id'].astype(str) + "-Bot" + df_agents_all['robot_id'].astype(str)

    return df_pos_all, df_agents_all

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data) # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - ci, mean + ci

def plot_heatmaps_side_by_side(df_pos1, df_pos2, label1, label2, save_path=None):
    if df_pos1 is None or df_pos2 is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Setup 1
    im1 = ax1.hist2d(df_pos1['x'], df_pos1['y'], bins=200, cmap='plasma', norm=mcolors.LogNorm(vmin=1))
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Position Heatmap - {label1} (Log Scale)')
    ax1.set_aspect('equal')

    # Setup 2
    im2 = ax2.hist2d(df_pos2['x'], df_pos2['y'], bins=200, cmap='plasma', norm=mcolors.LogNorm(vmin=1))
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'Position Heatmap - {label2} (Log Scale)')
    ax2.set_aspect('equal')

    # Single colorbar on the right
    fig.subplots_adjust(right=0.85, wspace=0)
    cbar = fig.colorbar(im2[3], ax=ax2, shrink=0.8, pad=0.02, label='Count (log scale)')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_distance_from_center_comparison(df_pos1, df_pos2, label1, label2, save_path=None):
    if df_pos1 is None or df_pos2 is None:
        return

    # Calculating arena center from combined data (this can be wrong if not many runs or unreached areas)
    all_x = pd.concat([df_pos1['x'], df_pos2['x']])
    all_y = pd.concat([df_pos1['y'], df_pos2['y']])
    arena_center_x = (all_x.min() + all_x.max()) / 2
    arena_center_y = (all_y.min() + all_y.max()) / 2

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'red']

    for i, (df_pos, label, color) in enumerate([(df_pos1, label1, colors[0]), (df_pos2, label2, colors[1])]):
        df_pos_copy = df_pos.copy()
        df_pos_copy['dist_from_center'] = np.sqrt((df_pos_copy['x'] - arena_center_x)**2 + (df_pos_copy['y'] - arena_center_y)**2)
        run_center_dist = df_pos_copy.groupby(['run_id', 'time'])['dist_from_center'].mean().reset_index()

        # Mean, CI, min, max per time
        stats_df = run_center_dist.groupby('time')['dist_from_center'].agg(list).reset_index()
        stats_df['mean'] = stats_df['dist_from_center'].apply(np.mean)
        stats_df['ci_lower'], stats_df['ci_upper'] = zip(*stats_df['dist_from_center'].apply(compute_confidence_interval))
        stats_df['min'] = stats_df['dist_from_center'].apply(np.min)
        stats_df['max'] = stats_df['dist_from_center'].apply(np.max)

        ax.plot(stats_df['time'], stats_df['min'], linestyle='--', color=color, alpha=0.7, label=f'{label} Min (run avg)')
        ax.plot(stats_df['time'], stats_df['max'], linestyle='--', color=color, alpha=0.7, label=f'{label} Max (run avg)')

        ax.plot(stats_df['time'], stats_df['mean'], linewidth=3, color=color, label=f'{label} Mean')
        ax.fill_between(stats_df['time'], stats_df['ci_lower'], stats_df['ci_upper'], alpha=0.3, color=color, label=f'{label} 95% CI')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance from Arena Center')
    ax.set_title('Distance from Arena Center Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_speed_comparison(df_pos1, df_pos2, label1, label2, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'red']

    for i, (df_pos, label, color) in enumerate([(df_pos1, label1, colors[0]), (df_pos2, label2, colors[1])]):
        # Calculate speed
        df_sorted = df_pos.sort_values(['run_id', 'robot_id', 'time'])
        df_sorted['dx'] = df_sorted.groupby(['run_id', 'robot_id'])['x'].diff()
        df_sorted['dy'] = df_sorted.groupby(['run_id', 'robot_id'])['y'].diff()
        df_sorted['dt'] = df_sorted.groupby(['run_id', 'robot_id'])['time'].diff()
        df_sorted['speed'] = np.sqrt(df_sorted['dx']**2 + df_sorted['dy']**2) / df_sorted['dt']
        df_speed = df_sorted.dropna(subset=['speed'])

        # Average speed per run per time
        run_speed = df_speed.groupby(['run_id', 'time'])['speed'].mean().reset_index()

        # Mean, CI, min, max per time
        stats_df = run_speed.groupby('time')['speed'].agg(list).reset_index()
        stats_df['mean'] = stats_df['speed'].apply(np.mean)
        stats_df['ci_lower'], stats_df['ci_upper'] = zip(*stats_df['speed'].apply(compute_confidence_interval))
        stats_df['min'] = stats_df['speed'].apply(np.min)
        stats_df['max'] = stats_df['speed'].apply(np.max)

        ax.plot(stats_df['time'], stats_df['min'], linestyle='--', color=color, alpha=0.7, label=f'{label} Min (run avg)')
        ax.plot(stats_df['time'], stats_df['max'], linestyle='--', color=color, alpha=0.7, label=f'{label} Max (run avg)')

        ax.plot(stats_df['time'], stats_df['mean'], linewidth=3, color=color, label=f'{label} Mean')
        ax.fill_between(stats_df['time'], stats_df['ci_lower'], stats_df['ci_upper'], alpha=0.3, color=color, label=f'{label} 95% CI')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed')
    ax.set_title('Speed Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_activity_comparison(df_agents1, df_agents2, label1, label2, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'red']

    for i, (df_agents, label, color) in enumerate([(df_agents1, label1, colors[0]), (df_agents2, label2, colors[1])]):
        # Mean per run per time
        run_activity = df_agents.groupby(['run_id', 'time'])['is_active'].mean().reset_index()

        # Mean, CI, min, max per time
        stats_df = run_activity.groupby('time')['is_active'].agg(list).reset_index()
        stats_df['mean'] = stats_df['is_active'].apply(np.mean)
        stats_df['ci_lower'], stats_df['ci_upper'] = zip(*stats_df['is_active'].apply(compute_confidence_interval))
        stats_df['min'] = stats_df['is_active'].apply(np.min)
        stats_df['max'] = stats_df['is_active'].apply(np.max)

        ax.plot(stats_df['time'], stats_df['min'], linestyle='--', color=color, alpha=0.7, label=f'{label} Min (run avg)')
        ax.plot(stats_df['time'], stats_df['max'], linestyle='--', color=color, alpha=0.7, label=f'{label} Max (run avg)')

        ax.plot(stats_df['time'], stats_df['mean'], linewidth=3, color=color, label=f'{label} Mean')
        ax.fill_between(stats_df['time'], stats_df['ci_lower'], stats_df['ci_upper'], alpha=0.3, color=color, label=f'{label} 95% CI')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activity Rate')
    ax.set_title('Activity Over Time')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_setups.py <runs_dir1> <runs_dir2>")
        sys.exit(1)

    runs_dir1 = sys.argv[1]
    runs_dir2 = sys.argv[2]

    label1 = os.path.basename(runs_dir1.rstrip('/'))
    label2 = os.path.basename(runs_dir2.rstrip('/'))

    # Load data
    df_pos1, df_agents1 = load_all_runs(runs_dir1)
    df_pos2, df_agents2 = load_all_runs(runs_dir2)
    
    if (df_pos1 is None and df_agents1 is None) or (df_pos2 is None and df_agents2 is None):
        print("No data loaded. Exiting.")
        return

    # Create plots directory
    os.makedirs('plots_comparison', exist_ok=True)

    print("Generating comparative plots...")

    plot_heatmaps_side_by_side(df_pos1, df_pos2, label1, label2, 'plots_comparison/heatmaps_side_by_side.png')
    plot_distance_from_center_comparison(df_pos1, df_pos2, label1, label2, 'plots_comparison/distance_from_center.png')
    plot_speed_comparison(df_pos1, df_pos2, label1, label2, 'plots_comparison/speed_over_time.png')
    plot_activity_comparison(df_agents1, df_agents2, label1, label2, 'plots_comparison/activity_over_time.png')

    print("\nComparative plots saved to 'plots_comparison/' directory")

if __name__ == '__main__':
    main()