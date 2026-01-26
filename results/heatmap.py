from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pandas as pd

base = Path(__file__).resolve().parent
file_path = base / "test_medeabatch" / "result_disk_100_120.0_2000.0.feather"

df = pd.read_feather(file_path)

plt.figure(figsize=(6,6))
plt.hist2d(df["x"], df["y"], bins=250, cmap="inferno", norm=LogNorm())
plt.colorbar(label="Passages (log)")
plt.gca().set_aspect("equal", "box")
plt.title("Heatmap de trajectoires")
plt.xlabel("x")
plt.ylabel("y")
plt.show()