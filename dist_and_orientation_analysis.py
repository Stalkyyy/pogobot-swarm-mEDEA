#!/usr/bin/env python3
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
import sys
from matplotlib.colors import LogNorm
from matplotlib import ticker

def load_single_run(run_path):
    if not os.path.isdir(run_path):
        print(f"Run folder {run_path} does not exist")
        return None, None

    all_pos_list = []
    all_agents_list = []
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
    
    return df_pos_all, df_agents_all


def filter_active(df, df_agents): # Get only active robots
    if df_agents is not None:
        df_agents_active = df_agents[['robot_id', 'generation', 'is_active']].drop_duplicates()
        df = df.merge(df_agents_active, on=['robot_id', 'generation'], how='left')
        return df[df['is_active'] == 1]
    return df

def filter_inactive(df, df_agents): # Get only inactive robots
    if df_agents is not None:
        df_agents_inactive = df_agents[['robot_id', 'generation', 'is_active']].drop_duplicates()
        df = df.merge(df_agents_inactive, on=['robot_id', 'generation'], how='left')
        return df[df['is_active'] == 0]
    return df


def plot_global_orientation_heatmap(df_pos, df_agents=None, robot_status='all', save_path=None, time_per_generation=6.6667, num_generations=None):
    if df_pos is None: return

    df = df_pos.copy()
    # Normalize to -pi to pi (not necessary but)
    df['angle'] = (df['angle'] + np.pi) % (2 * np.pi) - np.pi

    if robot_status == 'active':
        df = filter_active(df, df_agents)
    elif robot_status == 'inactive':
        df = filter_inactive(df, df_agents)

    # Binning
    num_generations = df['generation'].max() - df['generation'].min() + 1
    time_bins = np.linspace(df['time'].min(), df['time'].max(), num_generations + 1)
    angle_bins = np.linspace(-np.pi, np.pi, 31)

    fig, ax = plt.subplots(figsize=(12, 6))
    H, xedges, yedges, im = plt.hist2d(df['time'], df['angle'], bins=[time_bins, angle_bins], cmap='BuPu', norm=LogNorm())
    
    ax.set_facecolor('white')
    ax.set_xlabel(f'# timesteps (~{num_generations} generations)', fontsize=14)
    ax.set_ylabel('Global Orientation', fontsize=14)
    
    title = 'Global Orientation (NSEW)'
    if robot_status == 'active': title += ' - Active Robots'
    elif robot_status == 'inactive': title += ' - Inactive Robots'
    ax.set_title(title, fontsize=16)
    
    ax.grid(True, axis='x', color='darkgrey', alpha=0.3, zorder=1)

    # Global Compass Ticks
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(['West', 'South', 'East', 'North', 'West'], fontsize=12)

    cbar = plt.colorbar()
    cbar.formatter = ticker.ScalarFormatter()
    cbar.update_ticks()
    cbar.set_label('Count', fontsize=14)
    cbar.ax.text(0.5, 1.05, f'max: {int(H.max())}', transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_distance_heatmap_both(df_pos, center_x, center_y, df_agents, save_path=None, time_per_generation=6.6667, num_generations=None):
    if df_pos is None or df_agents is None: return

    df_active = filter_active(df_pos.copy(), df_agents)
    df_inactive = filter_inactive(df_pos.copy(), df_agents)

    df_active['distance'] = np.sqrt((df_active['x'] - center_x)**2 + (df_active['y'] - center_y)**2)
    df_inactive['distance'] = np.sqrt((df_inactive['x'] - center_x)**2 + (df_inactive['y'] - center_y)**2)

    if num_generations is None:
        num_generations = df_pos['generation'].max() - df_pos['generation'].min() + 1
    time_bins = np.linspace(df_pos['time'].min(), df_pos['time'].max(), num_generations + 1)
    dist_bins = np.linspace(0, max(df_active['distance'].max(), df_inactive['distance'].max()), 31)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
    fig.suptitle('Distribution of Distance from Center - Active vs Inactive Robots', fontsize=18)
    plt.subplots_adjust(top=0.85)

    # Active
    H1, xedges, yedges, im1 = ax1.hist2d(df_active['time'], df_active['distance'], bins=[time_bins, dist_bins], cmap='Blues', norm=LogNorm())
    ax1.set_facecolor('white')
    ax1.set_xlabel(f'# timesteps (~{num_generations} generations)', fontsize=14)
    ax1.set_ylabel('Distance to Center', fontsize=14)
    ax1.set_title('Active Robots', fontsize=16)
    ax1.grid(True, axis='x', color='darkgrey', alpha=0.3, zorder=1)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.formatter = ticker.ScalarFormatter()
    cbar1.update_ticks()
    cbar1.set_label('Count', fontsize=14)
    cbar1.ax.text(0.5, 1.05, f'max: {int(H1.max())}', transform=cbar1.ax.transAxes, ha='center', va='bottom', fontsize=12)

    # Inactive
    H2, xedges, yedges, im2 = ax2.hist2d(df_inactive['time'], df_inactive['distance'], bins=[time_bins, dist_bins], cmap='Reds', norm=LogNorm())
    ax2.set_facecolor('white')
    ax2.set_xlabel(f'# timesteps (~{num_generations} generations)', fontsize=14)
    ax2.set_ylabel('Distance to Center', fontsize=14)
    ax2.set_title('Inactive Robots', fontsize=16)
    ax2.grid(True, axis='x', color='darkgrey', alpha=0.3, zorder=1)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.formatter = ticker.ScalarFormatter()
    cbar2.update_ticks()
    cbar2.set_label('Count', fontsize=14)
    cbar2.ax.text(0.5, 1.05, f'max: {int(H2.max())}', transform=cbar2.ax.transAxes, ha='center', va='bottom', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_relative_orientation_heatmap(df_pos, df_agents=None, robot_status='all', center_x=0, center_y=0, save_path=None, time_per_generation=6.6667, num_generations=None):
    """
    Plot heatmap of generation vs Relative Orientation to Center.
    0 = Facing Center, +/- Pi = Facing Away.
    """
    if df_pos is None: return

    df = df_pos.copy()
    
    # from robot to center
    bearing_to_center = np.arctan2(center_y - df['y'], center_x - df['x'])
    
    relative_angle = bearing_to_center - df['angle']
    
    df['rel_angle'] = (relative_angle + np.pi) % (2 * np.pi) - np.pi
    
    if robot_status == 'active':
        df = filter_active(df, df_agents)
    elif robot_status == 'inactive':
        df = filter_inactive(df, df_agents)

    if num_generations is None:
        num_generations = df['generation'].max() - df['generation'].min() + 1
    time_bins = np.linspace(df['time'].min(), df['time'].max(), num_generations + 1)
    angle_bins = np.linspace(-np.pi, np.pi, 31)

    fig, ax = plt.subplots(figsize=(12, 6))
    H, xedges, yedges, im = plt.hist2d(df['time'], df['rel_angle'], bins=[time_bins, angle_bins], cmap='BuPu', norm=LogNorm())
    
    ax.set_facecolor('white')
    ax.set_xlabel(f'# timesteps (~{num_generations} generations)', fontsize=14)
    ax.set_ylabel('Orientation Relative to Center', fontsize=14)
    
    title = 'Robot Orientation Relative to Arena Center'
    if robot_status == 'active': title += ' - Active Robots'
    elif robot_status == 'inactive': title += ' - Inactive Robots'
    ax.set_title(title, fontsize=16)
    
    ax.grid(True, axis='x', color='darkgrey', alpha=0.3, zorder=1)

    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([
        'Facing Away', 
        'Right Shoulder to Center', 
        'Facing Center', 
        'Left Shoulder to Center', 
        'Facing Away'
    ], fontsize=12)

    cbar = plt.colorbar()
    cbar.formatter = ticker.ScalarFormatter()
    cbar.update_ticks()
    cbar.set_label('Count', fontsize=14)
    cbar.ax.text(0.5, 1.05, f'max: {int(H.max())}', transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_distance_heatmap(df_pos, center_x, center_y, df_agents=None, robot_status='all', save_path=None, time_per_generation=6.6667, num_generations=None):
    if df_pos is None: return

    df = df_pos.copy()
    df['distance'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)

    if robot_status == 'active':
        df = filter_active(df, df_agents)
    elif robot_status == 'inactive':
        df = filter_inactive(df, df_agents)

    if num_generations is None:
        num_generations = df['generation'].max() - df['generation'].min() + 1
    time_bins = np.linspace(df['time'].min(), df['time'].max(), num_generations + 1)
    dist_bins = np.linspace(0, df['distance'].max(), 31)

    fig, ax = plt.subplots(figsize=(12, 6))
    H, xedges, yedges, im = plt.hist2d(df['time'], df['distance'], bins=[time_bins, dist_bins], cmap='BuPu', norm=LogNorm())
    
    ax.set_facecolor('white')
    ax.set_xlabel(f'# timesteps (~{num_generations} generations)', fontsize=14)
    ax.set_ylabel('Distance to Center', fontsize=14)
    
    title = 'Distribution of Distance from Center'
    if robot_status == 'active': title += ' - Active Robots'
    elif robot_status == 'inactive': title += ' - Inactive Robots'
    ax.set_title(title, fontsize=16)
    
    ax.grid(True, axis='x', color='darkgrey', alpha=0.3, zorder=1)

    cbar = plt.colorbar()
    cbar.formatter = ticker.ScalarFormatter()
    cbar.update_ticks()
    cbar.set_label('Count', fontsize=14)
    cbar.ax.text(0.5, 1.05, f'max: {int(H.max())}', transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python single_run_analysis.py <data_folder>")
        sys.exit(1)

    run_path = sys.argv[1]
    df_pos, df_agents = load_single_run(run_path)

    if df_pos is None:
        print("No data loaded. Exiting.")
        return

    # Simulation parameters
    if df_agents is not None and not df_agents.empty:
        max_gen = df_agents['generation'].max()
        time_per_generation = df_pos['time'].max() / max_gen
        num_generations = max_gen
        print("Using data from agent logs to determine simulation parameters.")
    else:
        hz = 60  # Simulation frequency in Hz
        timesteps_per_generation = 400  # Timesteps per generation
        time_per_generation = timesteps_per_generation / hz
        num_generations = df_pos['generation'].max() - df_pos['generation'].min() + 1
        print("Assuming simulation at 60 Hz, 400 timesteps per generation, arena center auto-calculated. These are suppositions.")

    arena_center_x = None  # None for auto-calculation
    arena_center_y = None  # None for auto-calculation

    # Add generation column to df_pos
    df_pos['generation'] = (df_pos['time'] / time_per_generation).round().astype(int)

    if arena_center_x is None or arena_center_y is None:
        center_x = (df_pos['x'].min() + df_pos['x'].max()) / 2
        center_y = (df_pos['y'].min() + df_pos['y'].max()) / 2
        print(f"[DYNAMIC ARENA CENTER CALCULATION], can be wrong with low amount of data, set manually if necessary")
        print(f"Arena center calculated at: ({center_x:.1f}, {center_y:.1f})")
    else:
        center_x = arena_center_x
        center_y = arena_center_y
        print(f"Arena center set to: ({center_x:.1f}, {center_y:.1f})")

    directory = "distance_and_orientation_plots"
    os.makedirs(directory, exist_ok=True)
    print("Generating plots...")

    # plot_global_orientation_heatmap(df_pos, active_only=False, save_path=os.path.join(directory, 'orientation_global_all.png'), time_per_generation=time_per_generation, num_generations=num_generations)
    # plot_global_orientation_heatmap(df_pos, df_agents, active_only=True, save_path=os.path.join(directory, 'orientation_global_active.png'), time_per_generation=time_per_generation, num_generations=num_generations)
    # plot_relative_orientation_heatmap(df_pos, center_x=center_x, center_y=center_y, active_only=False, save_path=os.path.join(directory, 'orientation_relative_all.png'), time_per_generation=time_per_generation, num_generations=num_generations)
    plot_relative_orientation_heatmap(df_pos, df_agents, center_x=center_x, center_y=center_y, robot_status='active', save_path=os.path.join(directory, 'orientation_relative_active.png'), time_per_generation=time_per_generation, num_generations=num_generations)

    plot_distance_heatmap(df_pos, center_x, center_y, save_path=os.path.join(directory, 'distance_heatmap.png'), time_per_generation=time_per_generation, num_generations=num_generations)
    plot_distance_heatmap(df_pos, center_x, center_y, df_agents, robot_status='active', save_path=os.path.join(directory, 'distance_heatmap_active.png'), time_per_generation=time_per_generation, num_generations=num_generations)
    plot_distance_heatmap(df_pos, center_x, center_y, df_agents, robot_status='inactive', save_path=os.path.join(directory, 'distance_heatmap_inactive.png'), time_per_generation=time_per_generation, num_generations=num_generations)
    plot_distance_heatmap_both(df_pos, center_x, center_y, df_agents, save_path=os.path.join(directory, 'distance_heatmap_both.png'), time_per_generation=time_per_generation, num_generations=num_generations)

    print(f"Plots saved to '{directory}/' directory")

if __name__ == '__main__':
    main()