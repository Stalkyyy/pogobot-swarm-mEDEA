#!/usr/bin/env python3


from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUNS_ROOT = Path("runs_out")
OUT_DIR = RUNS_ROOT / "_analysis2"

CONTACT_RADIUS = 15.0
WINDOW_S = 10.0
MIN_POINTS_FOR_CORR = 5


ENV_REMAP = {
    "circle_central_light": "A",
    "circle_central_object": "B",
    "circle_central_object_light_blink": "C",
    "four_objects": "D",
}

METRIC_LABELS = {
    "lcc_frac": "Largest connected cluster fraction (share of robots in the LCC)",
    "num_groups": "Number of groups (connected components)",
    "top_pair_contact_frac": "Top pair contact share (fraction of contact edges in the strongest pair)",
    "genomes_received_per_s": "Genomes received per second (global rate)",
    "isolation_events_per_s": "Isolation events per second (rate of inactivity increments)",
    "singleton_frac": "Singleton fraction (share of robots with zero contacts)",
    "dyadness": "Dyadness (top pair share − mean share of top 5 pairs)",
    "partner_entropy": "Contact-degree entropy (proxy for contact diversity)",
}



TIME_CANDIDATES = ["t", "time", "sim_time"]
ID_CANDIDATES = ["id", "robot_id", "agent_id"]
X_CANDIDATES = ["x", "pos_x", "position_x"]
Y_CANDIDATES = ["y", "pos_y", "position_y"]

# these come from your main.c export
ACTIVE_CANDIDATES = ["is_active"]
RX_CANDIDATES = ["total_genomes_received"]
TX_CANDIDATES = ["total_genomes_sent"]
INACT_CANDIDATES = ["total_generations_inactive"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < MIN_POINTS_FOR_CORR:
        return float("nan")
    a0 = x.iloc[:, 0].to_numpy(dtype=float)
    b0 = x.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(a0) < 1e-12 or np.nanstd(b0) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a0, b0)[0, 1])


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """counts: nonnegative ints, entropy in nats."""
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts[counts > 0].astype(float) / float(total)
    return float(-(p * np.log(p)).sum())


def iter_feathers(root: Path) -> Iterable[Tuple[str, int, Path]]:
    """
    yields (env, seed, feather_path)
    expects: runs_out/<env>/seed_0000/data.feather
    """
    for env_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        for seed_dir in sorted(env_dir.glob("seed_*/")):
            fp = seed_dir / "data.feather"
            if fp.exists():
                seed = int(seed_dir.name.split("_")[-1])
                yield env_dir.name, seed, fp


def windowize_time(df: pd.DataFrame, tcol: str, window_s: float) -> pd.Series:
    t = df[tcol].to_numpy(dtype=float)
    t0 = float(np.nanmin(t))
    w = np.floor((t - t0) / window_s).astype(int)
    return pd.Series(w, index=df.index, name="win")


def compute_contact_edges_for_window(
    sub: pd.DataFrame, xcol: str, ycol: str, radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      edges as (u,v) arrays (undirected, u < v)
    """
    pts = sub[[xcol, ycol]].to_numpy(dtype=float)
    ids = sub["__id__"].to_numpy(dtype=int)
    n = len(ids)
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    u_list = []
    v_list = []
    r2 = radius * radius
    for i in range(n):
        xi, yi = pts[i, 0], pts[i, 1]
        for j in range(i + 1, n):
            dx = pts[j, 0] - xi
            dy = pts[j, 1] - yi
            if dx * dx + dy * dy <= r2:
                u_list.append(ids[i])
                v_list.append(ids[j])

    if not u_list:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.array(u_list, dtype=int), np.array(v_list, dtype=int)


def connected_components(num_nodes: int, edges_u: np.ndarray, edges_v: np.ndarray, node_ids: np.ndarray) -> List[List[int]]:
    idx = {rid: k for k, rid in enumerate(node_ids)}
    parent = np.arange(num_nodes, dtype=int)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in zip(edges_u, edges_v):
        union(idx[u], idx[v])

    groups: Dict[int, List[int]] = {}
    for rid in node_ids:
        r = find(idx[rid])
        groups.setdefault(r, []).append(int(rid))
    return list(groups.values())


def compute_network_timeseries(df: pd.DataFrame, env: str, seed: int) -> pd.DataFrame:
    tcol = pick_col(df, TIME_CANDIDATES)
    idcol = pick_col(df, ID_CANDIDATES)
    xcol = pick_col(df, X_CANDIDATES)
    ycol = pick_col(df, Y_CANDIDATES)
    if tcol is None or idcol is None or xcol is None or ycol is None:
        raise RuntimeError(f"Need time,id,x,y. Have cols: {list(df.columns)}")

    rxcol = pick_col(df, RX_CANDIDATES)
    txcol = pick_col(df, TX_CANDIDATES)
    inactcol = pick_col(df, INACT_CANDIDATES)

    df = df.sort_values([tcol, idcol]).copy()
    df["__t__"] = df[tcol].astype(float)
    df["__id__"] = df[idcol].astype(int)

    df["win"] = windowize_time(df, "__t__", WINDOW_S)

    ticks = df.groupby("__t__", sort=False).size().index.to_numpy(dtype=float)

    per_tick = pd.DataFrame({"time": ticks}).set_index("time")

    if rxcol is not None:
        d = df[["__t__", "__id__", rxcol]].copy()
        d["rx_diff"] = d.groupby("__id__")[rxcol].diff().fillna(0.0)
        rx_per_tick = d.groupby("__t__")["rx_diff"].sum()
        dt = pd.Series(ticks).diff()
        dt.index = ticks
        dt = dt.bfill().replace(0, np.nan).ffill()
        per_tick["genomes_received_per_s"] = (rx_per_tick / dt).astype(float)
    else:
        per_tick["genomes_received_per_s"] = np.nan

    if txcol is not None:
        d = df[["__t__", "__id__", txcol]].copy()
        d["tx_diff"] = d.groupby("__id__")[txcol].diff().fillna(0.0)
        tx_per_tick = d.groupby("__t__")["tx_diff"].sum()
        dt = pd.Series(ticks).diff()
        dt.index = ticks
        dt = dt.bfill().replace(0, np.nan).ffill()
        per_tick["genomes_sent_per_s"] = (tx_per_tick / dt).astype(float)
    else:
        per_tick["genomes_sent_per_s"] = np.nan

    if inactcol is not None:
        d = df[["__t__", "__id__", inactcol]].copy()
        d["inact_diff"] = d.groupby("__id__")[inactcol].diff().fillna(0.0)
        inact_per_tick = d.groupby("__t__")["inact_diff"].sum()
        dt = pd.Series(ticks).diff()
        dt.index = ticks
        dt = dt.bfill().replace(0, np.nan).ffill()
        per_tick["isolation_events_per_s"] = (inact_per_tick / dt).astype(float)
    else:
        per_tick["isolation_events_per_s"] = np.nan

    win_rows = []
    prev_top_pair = None
    persistent_hits = 0
    persistent_total = 0

    for win, sub in df.groupby("win", sort=True):
        t_window = float(sub["__t__"].min())

        last_t = float(sub["__t__"].max())
        snap = sub[sub["__t__"] == last_t][["__id__", xcol, ycol]].drop_duplicates("__id__")

        node_ids = snap["__id__"].to_numpy(dtype=int)
        n = len(node_ids)

        u, v = compute_contact_edges_for_window(snap, xcol, ycol, CONTACT_RADIUS)
        m = len(u)

        if n == 0:
            continue

        deg = {int(r): 0 for r in node_ids}
        for uu, vv in zip(u, v):
            deg[int(uu)] += 1
            deg[int(vv)] += 1

        singletons = sum(1 for r in node_ids if deg[int(r)] == 0)
        singleton_frac = singletons / float(n)

        comps = connected_components(n, u, v, node_ids)
        comp_sizes = sorted([len(c) for c in comps], reverse=True)
        num_groups = len(comp_sizes)
        lcc = comp_sizes[0] if comp_sizes else 1
        lcc_frac = lcc / float(n)

        if m == 0:
            top_pair_contact_frac = 0.0
            dyadness = 0.0
            top_pair = None
            partner_entropy = float("nan")
        else:
            pairs = list(zip(np.minimum(u, v), np.maximum(u, v)))
            vals, counts = np.unique(np.array(pairs, dtype=int), axis=0, return_counts=True)
            order = np.argsort(counts)[::-1]
            counts_sorted = counts[order]
            vals_sorted = vals[order]
            top_pair = (int(vals_sorted[0, 0]), int(vals_sorted[0, 1]))
            top_pair_contact_frac = float(counts_sorted[0] / counts_sorted.sum())

            topk = min(5, len(counts_sorted))
            mean_top5 = float(np.mean(counts_sorted[:topk] / counts_sorted.sum()))
            dyadness = float(top_pair_contact_frac - mean_top5)

            deg_vals = np.array([deg[int(r)] for r in node_ids], dtype=int)
            _, cts = np.unique(deg_vals, return_counts=True)
            partner_entropy = shannon_entropy_from_counts(cts)

        if prev_top_pair is not None:
            persistent_total += 1
            if top_pair == prev_top_pair and top_pair is not None:
                persistent_hits += 1
        prev_top_pair = top_pair

        wmask = (per_tick.index >= t_window) & (per_tick.index < (t_window + WINDOW_S))
        rx_mean = float(per_tick.loc[wmask, "genomes_received_per_s"].mean()) if wmask.any() else float("nan")
        iso_mean = float(per_tick.loc[wmask, "isolation_events_per_s"].mean()) if wmask.any() else float("nan")

        win_rows.append(
            {
                "env": env,
                "seed": seed,
                "time": t_window,
                "n_robots": n,
                "num_groups": num_groups,
                "lcc_frac": lcc_frac,
                "singleton_frac": singleton_frac,
                "top_pair_contact_frac": top_pair_contact_frac,
                "dyadness": dyadness,
                "partner_entropy": partner_entropy,
                "rx_genome_entropy": float("nan"),
                "genomes_received_per_s": rx_mean,
                "isolation_events_per_s": iso_mean,
            }
        )

    dyad_persistence = float(persistent_hits / persistent_total) if persistent_total > 0 else float("nan")
    out = pd.DataFrame(win_rows)
    out["dyad_persistence"] = dyad_persistence
    return out


def barplot_env(values: pd.Series, title: str, ylabel: str) -> None:
    values = values.sort_values()
    plt.figure(figsize=(9, 4))
    plt.bar(values.index.tolist(), values.values)
    plt.xticks(rotation=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def main() -> None:
    ensure_dir(OUT_DIR)

    all_ts: List[pd.DataFrame] = []
    for env, seed, fp in iter_feathers(RUNS_ROOT):
        env = ENV_REMAP.get(env, env)  # A/B/C/D naming
        print(f"Reading: {fp}")
        df = pd.read_feather(fp)
        ts = compute_network_timeseries(df, env, seed)
        all_ts.append(ts)

    all_df = pd.concat(all_ts, ignore_index=True)

    out_all = OUT_DIR / "ALL_network_timeseries.csv"
    all_df.to_csv(out_all, index=False)
    print(f"Saved: {out_all}")

    rows = []
    for (env, seed), g in all_df.groupby(["env", "seed"], sort=True):
        rows.append(
            {
                "env": env,
                "seed": seed,
                "corr_lcc_frac__rx_per_s": corr_safe(g["lcc_frac"], g["genomes_received_per_s"]),
                "corr_top_pair__rx_per_s": corr_safe(g["top_pair_contact_frac"], g["genomes_received_per_s"]),
                "corr_num_groups__isolation_per_s": corr_safe(g["num_groups"], g["isolation_events_per_s"]),
                "corr_singleton_frac__isolation_per_s": corr_safe(g["singleton_frac"], g["isolation_events_per_s"]),
            }
        )
    per = pd.DataFrame(rows)
    per.to_csv(OUT_DIR / "per_env_seed_correlations.csv", index=False)
    print(f"Saved: {OUT_DIR / 'per_env_seed_correlations.csv'}")

    env_mean = per.drop(columns=["seed"]).groupby("env").mean(numeric_only=True)
    env_mean.to_csv(OUT_DIR / "per_env_correlations_mean.csv")
    print(f"Saved: {OUT_DIR / 'per_env_correlations_mean.csv'}")

    # ---- SHOW ONLY THESE FOUR (no saving) ----

    # 1) Correlation bars: LCC fraction vs genome receiving rate
    barplot_env(
        env_mean["corr_lcc_frac__rx_per_s"],
        "Correlation: largest connected cluster fraction vs genome receiving rate (mean over seeds)",
        "Correlation",
    )

    # 2) Correlation bars: number of groups vs isolation-event rate
    barplot_env(
        env_mean["corr_num_groups__isolation_per_s"],
        "Correlation: number of groups vs isolation-event rate (mean over seeds)",
        "Correlation",
    )

    # 3) Dyadness bars (NO mean±sd error bars; just mean per env over seeds)
    dy = all_df.groupby(["env", "seed"])["dyadness"].mean().reset_index()
    dy_mean = dy.groupby("env")["dyadness"].mean().sort_values(ascending=False)

    plt.figure(figsize=(9, 4))
    plt.bar(dy_mean.index.tolist(), dy_mean.values)
    plt.xticks(rotation=0)
    plt.title("Dyadness (how strongly one pair dominates contacts)")
    plt.ylabel(METRIC_LABELS["dyadness"])
    plt.tight_layout()
    plt.show()

    # 4) Top-pair contact share over time (env mean over seeds)
    plt.figure(figsize=(9, 4))
    for env, g in all_df.groupby("env", sort=True):
        m = g.groupby("time")["top_pair_contact_frac"].mean()
        plt.plot(m.index.to_numpy(), m.to_numpy(), label=f"Env {env}")
    plt.title("Top pair contact share over time (mean over seeds)")
    plt.xlabel("Time")
    plt.ylabel(METRIC_LABELS["top_pair_contact_frac"])
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
