
'''
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# ---------------------------------------------------------------------------
# 1 – Import the environment class you already defined in rl_agent.py
# ---------------------------------------------------------------------------
from rl_agent import ThreeSensorTimeEnv

SENSOR_COSTS = [10, 4, 1]  # ECG, PPG, Temp  (mA for a 5‑s window)

# ---------------------------------------------------------------------------
# 2 – Load Q‑table utility ----------------------------------------------------
State = tuple  # any length >=5
QTable = Dict[Tuple[State, int], float]


def load_q_table(csv_path: Path) -> QTable:
    """
    Read qtable CSV produced by rl_agent.py.

    Accepts either 7-column (without prev_arr) or 8-column (with prev_arr) format.
    """
    out: QTable = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)            # skip header
        for row in reader:
            if len(row) == 7:
                b, t, arr, bp, fev, act, qv = row
                state = (int(b), int(t), int(arr), int(bp), int(fev))
            elif len(row) == 8:
                b, t, arr, bp, fev, prev_arr, act, qv = row
                state = (
                    int(b),
                    int(t),
                    int(arr),
                    int(bp),
                    int(fev),
                    int(prev_arr),
                )
            else:
                raise ValueError(f"Unexpected row length {len(row)} in Q-table")

            out[(state, int(act))] = float(qv)
    return out


# ---------------------------------------------------------------------------
# 3 – Policy helpers ---------------------------------------------------------

def greedy_rl(Q: QTable, state: State) -> int:
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    return int(np.argmax(qvals))


def always_on(_: State) -> int:
    return 0b111  # ECG, PPG, Temp on


def periodic_ecg(state: State) -> int:
    """Turn ECG ON every 30 s (6 time‑steps at 5 s cadence)."""
    _, t, *_ = state
    return 0b100 if (t % 6) == 0 else 0b000


# ---------------------------------------------------------------------------
# 4 – Synthetic anomaly scenario --------------------------------------------

def synthetic_scenario(T: int) -> List[dict]:
    rng = np.random.default_rng(0)
    return [
        {
            "arr_flag": rng.choice([0, 1], p=[0.9, 0.1]),
            "bp_flag": rng.choice([0, 1], p=[0.7, 0.3]),
            "fever_flag": rng.choice([0, 1], p=[0.9, 0.1]),
        }
        for _ in range(T)
    ]


# ---------------------------------------------------------------------------
# 5 – Episode runner ---------------------------------------------------------

def run_episode(env: ThreeSensorTimeEnv, policy_fn, label: str) -> dict:
    """Execute one full episode and collect detection/energy stats."""
    state = env.reset()
    det_hits = det_total = energy_cost = 0

    while True:
        action = policy_fn(state)
        next_state, _, done, _ = env.step(action)

        # --- energy bookkeeping -------------------------------------------
        ecg_on = (action >> 2) & 1
        ppg_on = (action >> 1) & 1
        tmp_on = action & 1
        energy_cost += (
            SENSOR_COSTS[0] * ecg_on
            + SENSOR_COSTS[1] * ppg_on
            + SENSOR_COSTS[2] * tmp_on
        )

        # --- anomaly detection bookkeeping --------------------------------
        for flag, on in zip(
            [env.arr_flag, env.bp_flag, env.fever_flag],
            [ecg_on, ppg_on, tmp_on],
        ):
            if flag:
                det_total += 1
                if on:
                    det_hits += 1

        if done:
            break
        state = next_state

    detection_rate = det_hits / det_total * 100 if det_total else 0.0
    mAh = energy_cost * 5 / 3600          # convert mA·5 s → mAh
    return dict(label=label, detection=detection_rate, mAh=mAh)


# ---------------------------------------------------------------------------
# 6 – Main: run three policies ----------------------------------------------
if __name__ == "__main__":
    STEPS = 12_000                       # ≈16 h at 5‑s cadence
    scenario = synthetic_scenario(STEPS)

    # Build environment factory so battery resets each run
    def make_env():
        return ThreeSensorTimeEnv(
            scenario,
            sensor_costs=SENSOR_COSTS,
            alpha=4.0,
            beta=0.008,
            max_battery=400_000,
            max_time_steps=STEPS,
        )

    Q = load_q_table(Path("qtable_sensors_time_ff.csv"))

    policies = [
        (lambda s: greedy_rl(Q, s), "RL"),
        (always_on, "Always‑on"),
        (periodic_ecg, "Periodic‑5/30"),
    ]

    results = []
    for fn, name in policies:
        env = make_env()
        results.append(run_episode(env, fn, name))

    print(f"\nDetection vs. Energy (synthetic {STEPS}‑step run)")
    print("Label           Det. (%)   Energy (mAh)")
    for res in results:
        print(f"{res['label']:<15s} {res['detection']:7.1f}      {res['mAh']:8.2f}")'''


# run_policies.py  –  updated for prevalence sweep & plotting
# --------------------------------------------------------------------------
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# --------------------------------------------------------------------------
# 1 – Environment and sensor‑cost setup
# --------------------------------------------------------------------------
from rl_agent import ThreeSensorTimeEnv   # <-- your existing class

SENSOR_COSTS = [10, 4, 1]                 # ECG, PPG, Temp  (mA per 5‑s slot)
STEPS = 12_000                            # ≈16 h at 5‑second cadence

# --------------------------------------------------------------------------
# 2 – Q‑table loader (unchanged)
# --------------------------------------------------------------------------
State = tuple
QTable = Dict[Tuple[State, int], float]


def load_q_table(csv_path: Path) -> QTable:
    """Load a Q‑table CSV of either 7‑ or 8‑column format."""
    out: QTable = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) == 7:
                b, t, arr, bp, fev, act, qv = row
                state = (int(b), int(t), int(arr), int(bp), int(fev))
            elif len(row) == 8:
                b, t, arr, bp, fev, prev_arr, act, qv = row
                state = (
                    int(b),
                    int(t),
                    int(arr),
                    int(bp),
                    int(fev),
                    int(prev_arr),
                )
            else:
                raise ValueError(f"Unexpected row length {len(row)}")
            out[(state, int(act))] = float(qv)
    return out


# --------------------------------------------------------------------------
# 3 – Policy helpers (unchanged)
# --------------------------------------------------------------------------
def greedy_rl(Q: QTable, state: State) -> int:
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    return int(np.argmax(qvals))


def always_on(_: State) -> int:
    return 0b111  # ECG, PPG, Temp ON


def periodic_ecg(state: State) -> int:
    """Turn ECG ON for one slot every 30 s (6 × 5 s)."""
    _, t, *_ = state
    return 0b100 if (t % 6) == 0 else 0b000


# --------------------------------------------------------------------------
# 4 – Synthetic anomaly generator  (now parameterised)
# --------------------------------------------------------------------------
def synthetic_scenario(
    T: int,
    p_arr: float = 0.10,
    p_bp: float = 0.30,
    p_fev: float = 0.10,
    rng_seed: int = 0,
) -> List[dict]:
    """Return a list of dicts, one per 5‑s window."""
    rng = np.random.default_rng(rng_seed)
    return [
        {
            "arr_flag": rng.random() < p_arr,
            "bp_flag": rng.random() < p_bp,
            "fever_flag": rng.random() < p_fev,
        }
        for _ in range(T)
    ]


# --------------------------------------------------------------------------
# 5 – Episode runner  (unchanged)
# --------------------------------------------------------------------------
def run_episode(env: ThreeSensorTimeEnv, policy_fn, label: str) -> dict:
    state = env.reset()
    det_hits = det_total = energy_cost = 0

    while True:
        action = policy_fn(state)
        next_state, _, done, _ = env.step(action)

        # energy bookkeeping
        ecg_on, ppg_on, tmp_on = (action >> 2) & 1, (action >> 1) & 1, action & 1
        energy_cost += (
            SENSOR_COSTS[0] * ecg_on
            + SENSOR_COSTS[1] * ppg_on
            + SENSOR_COSTS[2] * tmp_on
        )

        # detection bookkeeping
        for flag, on in zip(
            [env.arr_flag, env.bp_flag, env.fever_flag], [ecg_on, ppg_on, tmp_on]
        ):
            if flag:
                det_total += 1
                if on:
                    det_hits += 1

        if done:
            break
        state = next_state

    detection_rate = det_hits / det_total * 100 if det_total else 0.0
    mAh = energy_cost * 5 / 3600  # mA·s → mAh
    return dict(label=label, detection=detection_rate, mAh=mAh)


# --------------------------------------------------------------------------
# 6 – Helper: evaluate RL over a prevalence sweep
# --------------------------------------------------------------------------
def eval_prevalence(
    q_table: QTable,
    p_arr: float,
    p_bp: float = 0.30,
    p_fev: float = 0.10,
    seeds: int = 10,
) -> dict:
    recalls, energies = [], []
    for sd in range(seeds):
        scenario = synthetic_scenario(STEPS, p_arr, p_bp, p_fev, rng_seed=sd)
        env = ThreeSensorTimeEnv(
            scenario,
            sensor_costs=SENSOR_COSTS,
            alpha=12.0,
            beta=0.008,
            max_battery=400_000,
            max_time_steps=STEPS,
        )
        res = run_episode(env, lambda s: greedy_rl(q_table, s), "RL")
        recalls.append(res["detection"])
        energies.append(res["mAh"])
    return {
        "prev_arr": p_arr * 100,                       # %
        "recall_mean": np.mean(recalls),
        "recall_sd": np.std(recalls),
        "energy_mean": np.mean(energies),
        "energy_sd": np.std(energies),
    }


# --------------------------------------------------------------------------
# 7 – Main block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    Q = load_q_table(Path("qtable_sensors_time_ff.csv"))

    # -------- baseline comparison on default 10 % prevalence --------------
    scenario_default = synthetic_scenario(STEPS)  # 10 % arr, 30 % bp, 10 % fev

    def make_env(trace):
        return ThreeSensorTimeEnv(
            trace,
            sensor_costs=SENSOR_COSTS,
            alpha=12.0,
            beta=0.008,
            max_battery=400_000,
            max_time_steps=STEPS,
        )

    policies = [
        (lambda s: greedy_rl(Q, s), "RL"),
        (always_on, "Always‑on"),
        (periodic_ecg, "Periodic‑5/30"),
    ]

    print(f"\nDetection vs. Energy (synthetic {STEPS}‑step run, default prevalence)")
    print("Label           Det. (%)   Energy (mAh)")
    for fn, name in policies:
        env = make_env(scenario_default)
        res = run_episode(env, fn, name)
        print(f"{name:<15s} {res['detection']:7.1f}      {res['mAh']:8.2f}")

    # -------- prevalence sweep for the RL policy --------------------------
    prevalence_levels = [0.02, 0.05, 0.10, 0.15]         # 2 %, 5 %, 10 %, 15 %
    sweep = [eval_prevalence(Q, p) for p in prevalence_levels]

    print("\nDetection vs. Arrhythmia prevalence (RL, 10 seeds)")
    print("Prev (%)  Recall  (±SD)   Energy (mAh ±SD)")
    for r in sweep:
        print(f"{r['prev_arr']:>6.1f}   "
              f"{r['recall_mean']:5.1f} ±{r['recall_sd']:.1f}    "
              f"{r['energy_mean']:6.1f} ±{r['energy_sd']:.1f}")

    # save CSV for Table 5b
    csv_out = Path("rl_prevalence_sweep.csv")
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(sweep[0].keys())
        writer.writerows(row.values() for row in sweep)
    print(f"\nCSV written → {csv_out}")

    # optional plot (Fig. 6)
    try:
        import matplotlib.pyplot as plt

        plt.errorbar(
            [row["prev_arr"] for row in sweep],
            [row["recall_mean"] for row in sweep],
            yerr=[row["recall_sd"] for row in sweep],
            marker="o",
            capsize=4,
        )
        plt.xlabel("Arrhythmia prevalence (%)")
        plt.ylabel("Recall (%)")
        plt.title("Detection vs. Arrhythmia Prevalence (RL policy)")
        plt.grid(True)
        plt.tight_layout()
        fig_path = Path("fig_detection_prevalence.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved → {fig_path}")
    except ImportError:
        print("matplotlib not installed; skipping plot")
