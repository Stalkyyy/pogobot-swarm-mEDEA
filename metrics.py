import sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python metrics.py <run_folder>")
    sys.exit(1)

run_dir = Path(sys.argv[1])
data_file = run_dir / "data.feather"

if not data_file.exists():
    raise FileNotFoundError(f"{data_file} not found")

df = pd.read_feather(data_file)

# ---- BASIC SANITY ----
robots = df["robot_id"].nunique()
t_max = df["time"].max()

# ---- ACTIVE RATIO ----
if "is_active" in df.columns:
    active_ratio = df.groupby("time")["is_active"].mean()
    active_ratio_min = active_ratio.min()
    active_ratio_mean = active_ratio.mean()
    active_ratio_final = active_ratio.iloc[-1]
else:
    active_ratio_min = active_ratio_mean = active_ratio_final = np.nan

# ---- GENERATION ----
generation_counter_final_mean = (
    df.groupby("robot_id")["generation_counter"].max().mean()
    if "generation_counter" in df.columns else np.nan
)

genome_age_final_mean = (
    df.groupby("robot_id")["genome_age"].max().mean()
    if "genome_age" in df.columns else np.nan
)

genome_list_size_final_mean = (
    df.groupby("robot_id")["genome_list_size"].mean().iloc[-1]
    if "genome_list_size" in df.columns else np.nan
)

total_genomes_received_final_mean = (
    df.groupby("robot_id")["total_genomes_received"].max().mean()
    if "total_genomes_received" in df.columns else np.nan
)

total_generations_inactive_final_mean = (
    df.groupby("robot_id")["total_generations_inactive"].max().mean()
    if "total_generations_inactive" in df.columns else np.nan
)

sigma_final_mean = (
    df.groupby("robot_id")["sigma"].mean().iloc[-1]
    if "sigma" in df.columns else np.nan
)

# ---- MOTION METRICS ----
# Mean speed
if {"x", "y", "time"}.issubset(df.columns):
    df_sorted = df.sort_values(["robot_id", "time"])
    dx = df_sorted.groupby("robot_id")["x"].diff()
    dy = df_sorted.groupby("robot_id")["y"].diff()
    dt = df_sorted.groupby("robot_id")["time"].diff()
    speed = np.sqrt(dx**2 + dy**2) / dt
    mean_speed = speed.replace([np.inf, -np.inf], np.nan).mean()
else:
    mean_speed = np.nan

# Polarization (heading alignment)
if "angle" in df.columns:
    pol = df.groupby("time")["angle"].apply(
        lambda a: np.sqrt(np.mean(np.cos(a))**2 + np.mean(np.sin(a))**2)
    )
    polarization_mean = pol.mean()
else:
    polarization_mean = np.nan

# Mean nearest neighbor distance
if {"x", "y"}.issubset(df.columns):
    def mean_nnd(frame):
        pts = frame[["x", "y"]].values
        if len(pts) < 2:
            return np.nan
        dists = np.sqrt(((pts[:, None] - pts[None, :])**2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        return dists.min(axis=1).mean()

    mean_nearest_neighbor_dist = (
        df.groupby("time")
          .apply(mean_nnd)
          .mean()
    )
else:
    mean_nearest_neighbor_dist = np.nan

# ---- OUTPUT ----
print(f"run_dir: {run_dir.name}")
print(f"t_max: {t_max:.3f}")
print(f"robots: {robots}")

print(f"active_ratio_min: {active_ratio_min}")
print(f"active_ratio_mean: {active_ratio_mean}")
print(f"active_ratio_final: {active_ratio_final}")

print(f"generation_counter_final_mean: {generation_counter_final_mean}")
print(f"genome_age_final_mean: {genome_age_final_mean}")
print(f"genome_list_size_final_mean: {genome_list_size_final_mean}")
print(f"total_genomes_received_final_mean: {total_genomes_received_final_mean}")
print(f"total_generations_inactive_final_mean: {total_generations_inactive_final_mean}")
print(f"sigma_final_mean: {sigma_final_mean}")

print(f"mean_speed: {mean_speed}")
print(f"polarization_mean: {polarization_mean}")
print(f"mean_nearest_neighbor_dist: {mean_nearest_neighbor_dist}")
