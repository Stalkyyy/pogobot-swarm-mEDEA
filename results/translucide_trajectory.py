import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent
file_path = base / "test_medeabatch" / "result_disk_50_80.0_2000.0.feather"

number_of_robots = None   # Set to None to use every robots trajectory !!!
last_seconds = 100.0    # Set to None to use full time range !!!


df = pd.read_feather(file_path)

if number_of_robots is not None:
    number_of_robots = max(0, number_of_robots)
    number_of_robots = min(len(df["robot_id"].unique()), number_of_robots)
    robot_ids = df["robot_id"].unique()[:number_of_robots]
else :
    robot_ids = df["robot_id"].unique()

if last_seconds is not None:
    t_max = df["time"].max()
    df = df[df["time"] >= (t_max - last_seconds)]

plt.figure(figsize=(6,6))
for rid, g in df[df["robot_id"].isin(robot_ids)].groupby("robot_id"):
    plt.plot(g["x"], g["y"], alpha=0.25, linewidth=0.8)

plt.gca().set_aspect("equal", "box")
plt.title("Trajectoires (transparence)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
