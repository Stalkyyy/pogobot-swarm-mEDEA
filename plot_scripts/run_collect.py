#!/usr/bin/env python3


import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ===========================================================================================================

TIME_CANDIDATES = ["t", "time", "sim_time"]
ID_CANDIDATES = ["id", "robot_id", "agent_id"]

X_CANDIDATES = ["x", "pos_x", "position_x"]
Y_CANDIDATES = ["y", "pos_y", "position_y"]

THETA_CANDIDATES = ["theta", "angle", "orientation", "heading"]

ACTIVE_CANDIDATES = ["is_active"]
GENOME_LIST_CANDIDATES = ["genome_list_size"]
TOTAL_RX_CANDIDATES = ["total_genomes_received"]
TOTAL_TX_CANDIDATES = ["total_genomes_sent"]
TOTAL_INACT_GEN_CANDIDATES = ["total_generations_inactive"]

LAST_SENDER_CANDIDATES = ["last_rx_sender_id"]
LAST_HASH_CANDIDATES = ["last_rx_genome_hash"]
GEN_RX_UNIQUE_HASHES_CANDIDATES = ["gen_rx_unique_hashes"]

# ===========================================================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ===========================================================================================================

def parse_seeds(seed_args: List[str]) -> List[int]:
    """
    Accepts:
      --seeds 0 1 2
      --seeds 0-9
      --seeds 0-19 42 99
    """
    seeds: List[int] = []
    for token in seed_args:
        if "-" in token:
            a, b = token.split("-", 1)
            a_i, b_i = int(a), int(b)
            step = 1 if b_i >= a_i else -1
            seeds.extend(list(range(a_i, b_i + step, step)))
        else:
            seeds.append(int(token))

    # de-dupe preserve order
    out: List[int] = []
    seen = set()
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# ===========================================================================================================

def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)

# ===========================================================================================================

def patch_config(cfg: Dict, run_dir: Path, sim_time: Optional[float], seed: int) -> Dict[str, str]:
    """
    Patch config to write outputs into run_dir and set seed/sim_time.
    Only touches top-level keys; does not alter object definitions.
    """
    ensure_dir(run_dir)
    ensure_dir(run_dir / "frames")

    cfg["delete_old_files"] = True
    cfg["enable_data_logging"] = True
    cfg["enable_console_logging"] = True

    data_path = (run_dir / "data.feather").as_posix()
    console_path = (run_dir / "console.txt").as_posix()
    frames_name = (run_dir / "frames" / "f{:010.4f}.png").as_posix()

    cfg["data_filename"] = data_path
    cfg["console_filename"] = console_path
    cfg["frames_name"] = frames_name

    if sim_time is not None:
        cfg["simulation_time"] = float(sim_time)

    cfg["seed"] = int(seed)

    return {
        "data_filename": data_path,
        "console_filename": console_path,
        "frames_name": frames_name,
    }

# ===========================================================================================================

def run_sim(exe: Path, yaml_path: Path) -> None:
    cmd = [str(exe), "-c", str(yaml_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(exe.parent), check=True)

# ===========================================================================================================

def compute_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Returns:
      (timeseries_df, computed_metric_names, skipped_metric_reasons)
    """
    notes_ok: List[str] = []
    notes_skip: List[str] = []

    tcol = pick_col(df, TIME_CANDIDATES)
    idcol = pick_col(df, ID_CANDIDATES)
    if tcol is None or idcol is None:
        raise RuntimeError(f"Need time+id columns. Have: {list(df.columns)}")

    xcol = pick_col(df, X_CANDIDATES)
    ycol = pick_col(df, Y_CANDIDATES)
    thcol = pick_col(df, THETA_CANDIDATES)

    is_active = pick_col(df, ACTIVE_CANDIDATES)
    gls = pick_col(df, GENOME_LIST_CANDIDATES)
    total_rx = pick_col(df, TOTAL_RX_CANDIDATES)
    total_tx = pick_col(df, TOTAL_TX_CANDIDATES)
    total_inact = pick_col(df, TOTAL_INACT_GEN_CANDIDATES)

    last_sender = pick_col(df, LAST_SENDER_CANDIDATES)
    last_hash = pick_col(df, LAST_HASH_CANDIDATES)
    gen_rx_unique = pick_col(df, GEN_RX_UNIQUE_HASHES_CANDIDATES)

    df = df.sort_values([tcol, idcol]).copy()

    times = np.asarray(df[tcol].unique(), dtype=float)
    times.sort()

    out = pd.DataFrame({tcol: times}).set_index(tcol)

    dt = pd.Series(times, index=times).diff()
    if len(dt) > 1:
        dt.iloc[0] = dt.iloc[1]
    dt = dt.replace(0, np.nan).ffill().bfill()

    if is_active is not None:
        out["active_robots"] = df.groupby(tcol)[is_active].sum().reindex(times).to_numpy()
        notes_ok.append("active_robots")
    else:
        notes_skip.append("active_robots: missing is_active")

    if gls is not None:
        out["avg_genome_list_size"] = df.groupby(tcol)[gls].mean().reindex(times).to_numpy()
        notes_ok.append("avg_genome_list_size")
    else:
        notes_skip.append("avg_genome_list_size: missing genome_list_size")

    if total_rx is not None:
        d = df[[tcol, idcol, total_rx]].copy()
        d["rx_diff"] = d.groupby(idcol)[total_rx].diff().fillna(0.0)
        rx_per_tick = d.groupby(tcol)["rx_diff"].sum().reindex(times).fillna(0.0)
        out["genomes_received_per_s"] = (rx_per_tick.to_numpy() / dt.to_numpy()).astype(float)
        notes_ok.append("genomes_received_per_s")
    else:
        notes_skip.append("genomes_received_per_s: missing total_genomes_received")

    if total_tx is not None:
        d = df[[tcol, idcol, total_tx]].copy()
        d["tx_diff"] = d.groupby(idcol)[total_tx].diff().fillna(0.0)
        tx_per_tick = d.groupby(tcol)["tx_diff"].sum().reindex(times).fillna(0.0)
        out["genomes_sent_per_s"] = (tx_per_tick.to_numpy() / dt.to_numpy()).astype(float)
        notes_ok.append("genomes_sent_per_s")
    else:
        notes_skip.append("genomes_sent_per_s: missing total_genomes_sent")

    if total_inact is not None:
        d = df[[tcol, idcol, total_inact]].copy()
        d["inact_diff"] = d.groupby(idcol)[total_inact].diff().fillna(0.0)
        per_tick = d.groupby(tcol)["inact_diff"].sum().reindex(times).fillna(0.0)
        out["isolation_events_per_s"] = (per_tick.to_numpy() / dt.to_numpy()).astype(float)
        notes_ok.append("isolation_events_per_s")
    else:
        notes_skip.append("isolation_events_per_s: missing total_generations_inactive")

    if xcol is not None and ycol is not None:
        d = df[[tcol, idcol, xcol, ycol]].copy()
        d["dx"] = d.groupby(idcol)[xcol].diff()
        d["dy"] = d.groupby(idcol)[ycol].diff()
        d["dt"] = d.groupby(idcol)[tcol].diff()
        d = d[d["dt"] > 0].copy()
        d["speed"] = np.sqrt(d["dx"] ** 2 + d["dy"] ** 2) / d["dt"]
        out["mean_speed"] = d.groupby(tcol)["speed"].mean().reindex(times).to_numpy()
        notes_ok.append("mean_speed")
    else:
        notes_skip.append(f"mean_speed: missing pose columns (x={xcol}, y={ycol})")

    if thcol is not None:
        d = df[[tcol, thcol]].copy()
        c = np.cos(d[thcol].to_numpy(dtype=float))
        s = np.sin(d[thcol].to_numpy(dtype=float))
        d["c"] = c
        d["s"] = s
        mc = d.groupby(tcol)["c"].mean().reindex(times)
        ms = d.groupby(tcol)["s"].mean().reindex(times)
        out["alignment"] = np.sqrt(mc.to_numpy() ** 2 + ms.to_numpy() ** 2)  # 0..1
        notes_ok.append("alignment")
    else:
        notes_skip.append(f"alignment: missing heading column (theta={thcol})")

    if xcol is not None and ycol is not None:
        def nn_mean(sub: pd.DataFrame) -> float:
            pts = sub[[xcol, ycol]].to_numpy(dtype=float)
            n = pts.shape[0]
            if n < 2:
                return float("nan")
            mins = []
            for i in range(n):
                dx = pts[:, 0] - pts[i, 0]
                dy = pts[:, 1] - pts[i, 1]
                dist2 = dx * dx + dy * dy
                dist2[i] = np.inf
                mins.append(float(np.sqrt(dist2.min())))
            return float(np.mean(mins))

        out["mean_nearest_neighbor_distance"] = (
            df.groupby(tcol, sort=False).apply(nn_mean).reindex(times).to_numpy()
        )
        notes_ok.append("mean_nearest_neighbor_distance")
    else:
        notes_skip.append(f"mean_nearest_neighbor_distance: missing pose columns (x={xcol}, y={ycol})")

    if last_sender is not None:
        ent_list = []
        top_share_list = []
        idx_list = []
        for t, sub in df.groupby(tcol, sort=False):
            vals = sub[last_sender].to_numpy()
            # ignore defaults/unset
            vals = vals[(vals != 65535) & (vals != -1)]
            if vals.size == 0:
                ent_list.append(np.nan)
                top_share_list.append(np.nan)
                idx_list.append(float(t))
                continue
            _, cnt = np.unique(vals, return_counts=True)
            p = cnt / cnt.sum()
            ent = float(-(p * np.log(p)).sum())
            top_share = float(cnt.max() / cnt.sum())
            ent_list.append(ent)
            top_share_list.append(top_share)
            idx_list.append(float(t))

        s_ent = pd.Series(ent_list, index=idx_list).reindex(times)
        s_top = pd.Series(top_share_list, index=idx_list).reindex(times)

        out["rx_partner_entropy"] = s_ent.to_numpy()
        out["rx_top_sender_share"] = s_top.to_numpy()
        notes_ok.append("rx_partner_entropy")
        notes_ok.append("rx_top_sender_share")
    else:
        notes_skip.append("rx_partner_entropy/rx_top_sender_share: missing last_rx_sender_id")

    if last_hash is not None:
        ent_list = []
        dom_list = []
        idx_list = []
        for t, sub in df.groupby(tcol, sort=False):
            vals = sub[last_hash].to_numpy(dtype=np.int64)
            vals = vals[vals != 0]  # ignore unset
            if vals.size == 0:
                ent_list.append(np.nan)
                dom_list.append(np.nan)
                idx_list.append(float(t))
                continue
            _, cnt = np.unique(vals, return_counts=True)
            p = cnt / cnt.sum()
            ent = float(-(p * np.log(p)).sum())
            dom = float(cnt.max() / cnt.sum())
            ent_list.append(ent)
            dom_list.append(dom)
            idx_list.append(float(t))

        s_ent = pd.Series(ent_list, index=idx_list).reindex(times)
        s_dom = pd.Series(dom_list, index=idx_list).reindex(times)

        out["rx_genome_entropy"] = s_ent.to_numpy()
        out["rx_dominant_hash_share"] = s_dom.to_numpy()
        notes_ok.append("rx_genome_entropy")
        notes_ok.append("rx_dominant_hash_share")
    else:
        notes_skip.append("rx_genome_entropy/rx_dominant_hash_share: missing last_rx_genome_hash")

    # --- NEW: per-generation unique hashes proxy (requires gen_rx_unique_hashes) ---
    if gen_rx_unique is not None:
        out["avg_rx_unique_hashes_per_gen"] = df.groupby(tcol)[gen_rx_unique].mean().reindex(times).to_numpy()
        notes_ok.append("avg_rx_unique_hashes_per_gen")
    else:
        notes_skip.append("avg_rx_unique_hashes_per_gen: missing gen_rx_unique_hashes")

    return out.reset_index(), notes_ok, notes_skip


# ===========================================================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="Path to simulator binary (e.g. ./pogobot-swarm-mEDEA)")
    ap.add_argument("--configs", nargs="+", required=True, help="YAML configs to run")
    ap.add_argument("--out", required=True, help="Output directory root")
    ap.add_argument("--sim-time", type=float, default=None, help="Override simulation_time in seconds")
    ap.add_argument("--seeds", nargs="+", default=["0-4"], help="Seeds, e.g. 0-9 42 (default 0-4)")
    args = ap.parse_args()

    exe = Path(args.exe).resolve()
    if not exe.exists():
        print(f"ERROR: exe not found: {exe}")
        return 2

    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    seeds = parse_seeds(args.seeds)

    for cfg_path in args.configs:
        src = Path(cfg_path).resolve()
        if not src.exists():
            print(f"ERROR: config not found: {src}")
            return 2

        env_name = src.stem
        env_dir = out_root / env_name
        ensure_dir(env_dir)

        base_cfg = load_yaml(src)

        for seed in seeds:
            run_dir = env_dir / f"seed_{seed:04d}"
            ensure_dir(run_dir)

            cfg = dict(base_cfg)  # shallow copy is enough for top-level patches
            paths = patch_config(cfg, run_dir, args.sim_time, seed)

            patched_yaml = run_dir / "config_patched.yaml"
            save_yaml(cfg, patched_yaml)

            try:
                run_sim(exe, patched_yaml)
            except subprocess.CalledProcessError as e:
                (run_dir / "run_failed.txt").write_text(
                    f"Run failed.\nCommand: {exe} -c {patched_yaml}\nExit: {e.returncode}\n",
                    encoding="utf-8",
                )
                print(f"ERROR: run failed for {env_name} seed {seed}. See {run_dir/'run_failed.txt'}")
                return int(e.returncode) or 1

            data_file = Path(paths["data_filename"])
            if not data_file.exists():
                (run_dir / "run_failed.txt").write_text(
                    f"No data.feather produced at: {data_file}\n",
                    encoding="utf-8",
                )
                print(f"ERROR: no data.feather for {env_name} seed {seed}")
                return 1

            df = pd.read_feather(data_file)
            (run_dir / "columns.txt").write_text("\n".join(map(str, df.columns)), encoding="utf-8")

            notes_path = run_dir / "metrics_notes.txt"
            try:
                ts, ok, skip = compute_metrics(df)
                ts.to_csv(run_dir / "metrics_timeseries.csv", index=False)
                notes_path.write_text(
                    "COMPUTED:\n- " + "\n- ".join(ok) + "\n\nSKIPPED:\n- " + "\n- ".join(skip) + "\n",
                    encoding="utf-8",
                )
            except Exception as e:
                notes_path.write_text(f"Metrics failed: {e}\n", encoding="utf-8")

            info = {
                "env": env_name,
                "seed": seed,
                "source_config": str(src),
                "patched_config": str(patched_yaml),
                "data_file": str(data_file),
                "console_file": str(paths["console_filename"]),
                "sim_time_override": args.sim_time,
            }
            (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

            print(f"DONE: {env_name} seed {seed} -> {run_dir}")

    return 0


# ===========================================================================================================

if __name__ == "__main__":
    raise SystemExit(main())