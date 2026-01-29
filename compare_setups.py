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

# Set style and increase font sizes
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

# Global colors for plots
colors = ['red', 'green']

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

def plot_heatmaps_side_by_side(df_pos1, df_pos2, label1, label2, bins, save_path=None, num_runs=None, max_time=None, num_robots=None):
    if df_pos1 is None or df_pos2 is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Setup 1
    H1, xedges1, yedges1 = np.histogram2d(df_pos1['x'], -df_pos1['y'], bins=bins)
    H1 = H1.T
    masked_H1 = np.ma.masked_where(H1 == 0, H1)
    im1 = ax1.imshow(masked_H1, extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]], origin='lower', cmap='plasma', norm=mcolors.LogNorm())
    ax1.set_facecolor('black')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'{label1}')
    ax1.set_aspect('equal')
    ax1.grid(True, color='darkgrey', alpha=0.1, zorder=-1)

    # Setup 2
    H2, xedges2, yedges2 = np.histogram2d(df_pos2['x'], -df_pos2['y'], bins=bins)
    H2 = H2.T
    masked_H2 = np.ma.masked_where(H2 == 0, H2)
    im2 = ax2.imshow(masked_H2, extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], origin='lower', cmap='plasma', norm=mcolors.LogNorm())
    ax2.set_facecolor('black')
    ax2.set_xlabel('X Position')
    # Removing y-axis label for second plot (still got ticks)
    ax2.set_ylabel('')
    ax2.set_title(f'{label2}')
    ax2.set_aspect('equal')
    ax2.grid(True, color='darkgrey', alpha=0.1, zorder=-1)

    # Overall title
    title_info = f"Position Heatmaps Comparison - {num_runs} runs, {max_time}s, {num_robots} robots per run" if num_runs and max_time and num_robots else "Position Heatmaps Comparison"
    fig.suptitle(title_info, fontsize=20)

    # Single colorbar on the right
    fig.subplots_adjust(right=0.85, wspace=0)
    cbar = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02, label='Count (log scale)')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_distance_from_center_comparison(df_pos1, df_pos2, label1, label2, save_path=None, num_runs=None, max_time=None, num_robots=None):
    if df_pos1 is None or df_pos2 is None:
        return

    # Calculating arena center from combined data (this can be wrong if not many runs or unreached areas)
    all_x = pd.concat([df_pos1['x'], df_pos2['x']])
    all_y = pd.concat([df_pos1['y'], df_pos2['y']])
    arena_center_x = (all_x.min() + all_x.max()) / 2
    arena_center_y = (all_y.min() + all_y.max()) / 2

    fig, ax = plt.subplots(figsize=(15, 8))

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
    title_info = f"Distance from Arena Center Over Time - {num_runs} runs, {max_time}s, {num_robots} robots per run" if num_runs and max_time and num_robots else "Distance from Arena Center Over Time"
    ax.set_title(title_info)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_speed_comparison(df_pos1, df_pos2, label1, label2, save_path=None, num_runs=None, max_time=None, num_robots=None):
    fig, ax = plt.subplots(figsize=(15, 8))

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
    title_info = f"Speed Over Time - {num_runs} runs, {max_time}s, {num_robots} robots per run" if num_runs and max_time and num_robots else "Speed Over Time"
    ax.set_title(title_info)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_activity_comparison(df_agents1, df_agents2, label1, label2, save_path=None, num_runs=None, max_time=None, num_robots=None):
    fig, ax = plt.subplots(figsize=(15, 8))

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
    title_info = f"Activity Over Time - {num_runs} runs, {max_time}s, {num_robots} robots per run" if num_runs and max_time and num_robots else "Activity Over Time"
    ax.set_title(title_info)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_pairwise_distance_comparison(df_pos1, df_pos2, label1, label2, save_path=None, num_runs=None, max_time=None, num_robots=None):
    if df_pos1 is None or df_pos2 is None:
        return

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, (df_pos, label, color) in enumerate([(df_pos1, label1, colors[0]), (df_pos2, label2, colors[1])]):
        # For each run and time, compute mean pairwise distance
        def compute_mean_pairwise_dist(group):
            positions = group[['x', 'y']].values
            if len(positions) < 2:
                return np.nan
            distances = pdist(positions)
            return np.mean(distances)

        run_pairwise_dist = df_pos.groupby(['run_id', 'time']).apply(compute_mean_pairwise_dist, include_groups=False).reset_index(name='mean_pairwise_dist')

        # Mean, CI, min, max per time
        stats_df = run_pairwise_dist.groupby('time')['mean_pairwise_dist'].agg(list).reset_index()
        stats_df['mean'] = stats_df['mean_pairwise_dist'].apply(np.mean)
        stats_df['ci_lower'], stats_df['ci_upper'] = zip(*stats_df['mean_pairwise_dist'].apply(compute_confidence_interval))
        stats_df['min'] = stats_df['mean_pairwise_dist'].apply(np.min)
        stats_df['max'] = stats_df['mean_pairwise_dist'].apply(np.max)

        ax.plot(stats_df['time'], stats_df['min'], linestyle='--', color=color, alpha=0.7, label=f'{label} Min (run avg)')
        ax.plot(stats_df['time'], stats_df['max'], linestyle='--', color=color, alpha=0.7, label=f'{label} Max (run avg)')

        ax.plot(stats_df['time'], stats_df['mean'], linewidth=3, color=color, label=f'{label} Mean')
        ax.fill_between(stats_df['time'], stats_df['ci_lower'], stats_df['ci_upper'], alpha=0.3, color=color, label=f'{label} 95% CI')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Pairwise Distance')
    title_info = f"Mean Pairwise Distance Over Time - {num_runs} runs, {max_time}s, {num_robots} robots per run" if num_runs and max_time and num_robots else "Mean Pairwise Distance Over Time"
    ax.set_title(title_info)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_genome_polarization_comparison(df_pos1, df_pos2, label1, label2, save_path=None, num_runs=None, max_time=None, num_robots=None):
    if df_pos1 is None or df_pos2 is None:
        return

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, (df_pos, label, color) in enumerate([(df_pos1, label1, colors[0]), (df_pos2, label2, colors[1])]):
        # Calculate direction vectors
        df_sorted = df_pos.sort_values(['run_id', 'robot_id', 'time'])
        df_sorted['dx'] = df_sorted.groupby(['run_id', 'robot_id'])['x'].diff()
        df_sorted['dy'] = df_sorted.groupby(['run_id', 'robot_id'])['y'].diff()
        df_sorted = df_sorted.dropna(subset=['dx', 'dy'])

        # Compute polarization per run per time
        polarization_data = []
        for (run_id, time_val), group in df_sorted.groupby(['run_id', 'time']):
            directions = group[['dx', 'dy']].values
            speeds = np.linalg.norm(directions, axis=1)
            # Only consider robots with non-zero speed
            active_mask = speeds > 1e-6  # small threshold
            if np.sum(active_mask) == 0:
                pol = 0.0
            else:
                unit_dirs = directions[active_mask] / speeds[active_mask, np.newaxis]
                resultant = np.sum(unit_dirs, axis=0)
                pol = np.linalg.norm(resultant) / np.sum(active_mask)
            polarization_data.append({'run_id': run_id, 'time': time_val, 'polarization': pol})

        df_pol = pd.DataFrame(polarization_data)

        # Mean, CI, min, max per time
        stats_df = df_pol.groupby('time')['polarization'].agg(list).reset_index()
        stats_df['mean'] = stats_df['polarization'].apply(np.mean)
        stats_df['ci_lower'], stats_df['ci_upper'] = zip(*stats_df['polarization'].apply(compute_confidence_interval))
        stats_df['min'] = stats_df['polarization'].apply(np.min)
        stats_df['max'] = stats_df['polarization'].apply(np.max)

        ax.plot(stats_df['time'], stats_df['min'], linestyle='--', color=color, alpha=0.7, label=f'{label} Min (run avg)')
        ax.plot(stats_df['time'], stats_df['max'], linestyle='--', color=color, alpha=0.7, label=f'{label} Max (run avg)')

        ax.plot(stats_df['time'], stats_df['mean'], linewidth=3, color=color, label=f'{label} Mean')
        ax.fill_between(stats_df['time'], stats_df['ci_lower'], stats_df['ci_upper'], alpha=0.3, color=color, label=f'{label} 95% CI')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Genome Polarization')
    title_info = f"Genome Polarization Over Time - {num_runs} runs, {max_time}s, {num_robots} robots per run" if num_runs and max_time and num_robots else "Genome Polarization Over Time"
    ax.set_title(title_info)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_genomes_received_comparison(df_agents1, df_agents2, label1, label2, save_path=None, num_runs=None, max_time=None, num_robots=None):
    if df_agents1 is None or df_agents2 is None:
        return

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, (df_agents, label, color) in enumerate([(df_agents1, label1, colors[0]), (df_agents2, label2, colors[1])]):
        # Calculate non-cumulative genomes received
        df_sorted = df_agents.sort_values(['run_id', 'robot_id', 'time'])
        df_sorted['genomes_received'] = df_sorted.groupby(['run_id', 'robot_id'])['total_genomes_received'].diff().fillna(0)

        # Sum per run per time
        run_genomes = df_sorted.groupby(['run_id', 'time'])['genomes_received'].sum().reset_index()

        # Mean, CI, min, max per time
        stats_df = run_genomes.groupby('time')['genomes_received'].agg(list).reset_index()
        stats_df['mean'] = stats_df['genomes_received'].apply(np.mean)
        stats_df['ci_lower'], stats_df['ci_upper'] = zip(*stats_df['genomes_received'].apply(compute_confidence_interval))
        stats_df['min'] = stats_df['genomes_received'].apply(np.min)
        stats_df['max'] = stats_df['genomes_received'].apply(np.max)

        ax.plot(stats_df['time'], stats_df['min'], linestyle='--', color=color, alpha=0.7, label=f'{label} Min (run avg)')
        ax.plot(stats_df['time'], stats_df['max'], linestyle='--', color=color, alpha=0.7, label=f'{label} Max (run avg)')

        ax.plot(stats_df['time'], stats_df['mean'], linewidth=3, color=color, label=f'{label} Mean')
        ax.fill_between(stats_df['time'], stats_df['ci_lower'], stats_df['ci_upper'], alpha=0.3, color=color, label=f'{label} 95% CI')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of Genomes Received (Non-cumulative)')
    title_info = f"Genomes Received Over Time - {num_runs} runs, {max_time}s, {num_robots} robots per run" if num_runs and max_time and num_robots else "Genomes Received Over Time"
    ax.set_title(title_info)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python compare_setups.py <runs_dir1> <runs_dir2> [label1] [label2]")
        sys.exit(1)

    runs_dir1 = sys.argv[1]
    runs_dir2 = sys.argv[2]

    if len(sys.argv) >= 4:
        label1 = sys.argv[3]
    else:
        label1 = os.path.basename(runs_dir1.rstrip('/'))

    if len(sys.argv) >= 5:
        label2 = sys.argv[4]
    else:
        label2 = os.path.basename(runs_dir2.rstrip('/'))

    # Load data
    df_pos1, df_agents1 = load_all_runs(runs_dir1)
    df_pos2, df_agents2 = load_all_runs(runs_dir2)
    
    if (df_pos1 is None and df_agents1 is None) or (df_pos2 is None and df_agents2 is None):
        print("No data loaded. Exiting.")
        return
    print("Data loaded successfully.")

    # Compute stats
    num_runs1 = len([f for f in glob.glob(os.path.join(runs_dir1, "run_*")) if os.path.isdir(f)])
    num_runs2 = len([f for f in glob.glob(os.path.join(runs_dir2, "run_*")) if os.path.isdir(f)])
    max_time1 = df_pos1['time'].max() if df_pos1 is not None else 0
    max_time2 = df_pos2['time'].max() if df_pos2 is not None else 0
    num_robots1 = df_agents1['robot_id'].nunique() if df_agents1 is not None else 0
    num_robots2 = df_agents2['robot_id'].nunique() if df_agents2 is not None else 0

    # Assume same for both if not, but for title, use the values
    num_runs = num_runs1  # assuming same
    max_time = int(max(max_time1, max_time2))
    num_robots = max(num_robots1, num_robots2)

    # Create plots directory
    os.makedirs('plots_comparison', exist_ok=True)

    print("Generating comparative plots...")

    # Heatmaps at different resolutions
    for bins in [800, 400, 200, 100, 50, 20]:
        print(f"Generating heatmaps with {bins} bins...")
        plot_heatmaps_side_by_side(df_pos1, df_pos2, label1, label2, bins, f'plots_comparison/heatmaps_{bins}bins.png', num_runs, max_time, num_robots)

    plot_distance_from_center_comparison(df_pos1, df_pos2, label1, label2, 'plots_comparison/distance_from_center.png', num_runs, max_time, num_robots)
    plot_speed_comparison(df_pos1, df_pos2, label1, label2, 'plots_comparison/speed_over_time.png', num_runs, max_time, num_robots)
    plot_activity_comparison(df_agents1, df_agents2, label1, label2, 'plots_comparison/activity_over_time.png', num_runs, max_time, num_robots)
    plot_pairwise_distance_comparison(df_pos1, df_pos2, label1, label2, 'plots_comparison/pairwise_distance.png', num_runs, max_time, num_robots)
    plot_genome_polarization_comparison(df_pos1, df_pos2, label1, label2, 'plots_comparison/genome_polarization.png', num_runs, max_time, num_robots)
    plot_genomes_received_comparison(df_agents1, df_agents2, label1, label2, 'plots_comparison/genomes_received.png', num_runs, max_time, num_robots)

    print("\nComparative plots saved to 'plots_comparison/' directory")

if __name__ == '__main__':
    main()