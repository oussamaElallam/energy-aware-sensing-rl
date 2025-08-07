import random
import csv
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
class ThreeSensorTimeEnv:
    """
    Three-sensor RL environment (ECG, PPG, Temp).

    state = (battery_disc 0-10,
             time_bucket 0-59,         # 5 s windows modulo 5 min
             arr_flag, bp_flag, fever_flag)

    action ∈ {0..7} ⇒ 3-bit mask of which sensors are ON.
    reward = α·detection_success – β·energy_cost
    """

    def __init__(
        self,
        data: List[dict],
        sensor_costs,
        alpha: float = 15.0,
        beta: float = 0.008,
        lambda_risk: float = 0.0,
        max_battery: int = 400_000,     # mA·5 s units  (~500 mAh)
        max_time_steps: int = 12_000,   # 16 h at 5 s cadence
    ):
        self.data = data
        self.sensor_costs = sensor_costs          # [ECG, PPG, Temp] in mA/5 s
        self.alpha = alpha
        self.beta = beta
        self.lambda_risk = lambda_risk
        self.max_battery = max_battery
        self.max_time_steps = max_time_steps
        self.action_space = range(8)

        self.reset()

    # ── RL API ────────────────────────────────────────────────────────────
    def reset(self):
        self.t = 0
        self.battery = self.max_battery
        self.done = False

        f = self.data[0]
        self.arr_flag, self.bp_flag, self.fever_flag = (
            int(f["arr_flag"]),
            int(f["bp_flag"]),
            int(f["fever_flag"]),
        )
        return self._get_state()

    def step(self, action_int: int):
        ecg_on = (action_int >> 2) & 1
        ppg_on = (action_int >> 1) & 1
        tmp_on = action_int & 1

        # ─ energy cost ──────────────────────────────────────────────────────
        cost = (
                self.sensor_costs[0] * ecg_on +
                self.sensor_costs[1] * ppg_on +
                self.sensor_costs[2] * tmp_on
        )
        self.battery = max(0, self.battery - cost)

        # ─ detection success ────────────────────────────────────────────────
        success = (
                (self.arr_flag and ecg_on) +
                (self.bp_flag and ppg_on) +
                (self.fever_flag and tmp_on)
        )
        
        # ─ missed events (risk component) ───────────────────────────────────
        missed_events = (
                (self.arr_flag and not ecg_on) +
                (self.bp_flag and not ppg_on) +
                (self.fever_flag and not tmp_on)
        )
        
        reward = self.alpha * success - self.beta * cost - self.lambda_risk * missed_events

        # ─ advance time ─────────────────────────────────────────────────────
        self.t += 1
        if self.t >= self.max_time_steps or self.battery == 0:
            self.done = True
        else:
            # NEW ▸ remember *previous* arrhythmia flag for next state
            self.prev_arr = self.arr_flag

            f = self.data[self.t]
            self.arr_flag = int(f["arr_flag"])
            self.bp_flag = int(f["bp_flag"])
            self.fever_flag = int(f["fever_flag"])



        return self._get_state(), reward, self.done, {}

    # ── internal ──────────────────────────────────────────────────────────
    def _get_state(self):
        battery_disc = min(10, self.battery // 10)
        time_bucket = self.t % 60  # 5-s slots modulo 5 min
        return (
            int(battery_disc),
            int(time_bucket),
            self.arr_flag,
            self.bp_flag,
            self.fever_flag,
            getattr(self, "prev_arr", 0),  # NEW field (defaults to 0)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Q-learning
# ─────────────────────────────────────────────────────────────────────────────
State = Tuple[int, int, int, int, int]
QTable = Dict[Tuple[State, int], float]


def q_learning_train(
    env: ThreeSensorTimeEnv,
    episodes: int = 5_000,
    gamma: float = 0.95,
    alpha_lr: float = 0.1,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.998,
) -> Tuple[QTable, List[float]]:
    Q: QTable = {}
    rewards: List[float] = []

    for _ in range(episodes):
        s = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            # ε-greedy
            if random.random() < epsilon:
                a = random.randrange(8)
            else:
                a = int(
                    np.argmax([Q.get((s, b), 0.0) for b in range(8)])
                )

            s2, r, done, _ = env.step(a)

            best_next = max(Q.get((s2, b), 0.0) for b in range(8))
            td_target = r + gamma * best_next
            td_error = td_target - Q.get((s, a), 0.0)
            Q[(s, a)] = Q.get((s, a), 0.0) + alpha_lr * td_error

            s = s2
            ep_r += r

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(ep_r)

    return Q, rewards


def save_q_table_csv(Q: QTable, path: Path):
    path = Path(path)
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["battery", "t_bucket", "arr", "bp", "fever", "action", "q"]
        )
        for (state, a), v in Q.items():
            wr.writerow([*state, a, v])
    print(f"Q-table saved to {path} ({len(Q)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# Train if run as a script
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Q-learning agent for energy-aware sensing')
    parser.add_argument('--lambda_risk', type=float, default=0.0, 
                       help='Risk penalty weight for missed events (default: 0.0)')
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes (default: 3000)')
    parser.add_argument('--output', type=str, default='qtable_sensors_time.csv',
                       help='Output Q-table filename (default: qtable_sensors_time.csv)')
    args = parser.parse_args()
    
    STEPS = 12_000                     # 16 h at 5 s cadence
    rng = np.random.default_rng(0)
    scenario = [
        {
            "arr_flag":  rng.choice([0, 1], p=[0.7, 0.3]),
            "bp_flag":   rng.choice([0, 1], p=[0.6, 0.4]),
            "fever_flag": rng.choice([0, 1], p=[0.8, 0.2]),
        }
        for _ in range(STEPS)
    ]

    env = ThreeSensorTimeEnv(
        scenario,
        sensor_costs=[10, 4, 1],
        alpha=15.0,
        beta=0.008,
        lambda_risk=args.lambda_risk,
        max_battery=400_000,
        max_time_steps=STEPS,
    )

    Q, R = q_learning_train(env, episodes=args.episodes)
    print(f"Lambda_risk: {args.lambda_risk}")
    print(f"Avg reward (last 50 eps): {np.mean(R[-50:]):.2f}")

    # ────────── PROBE: how many state–actions prefer at least one sensor ON
    on_pref  = sum(1 for (s, a), v in Q.items() if v > 0 and a != 0)
    total_sa = len(Q)
    print(f"{on_pref/total_sa*100:.1f}% of state–actions favour ≥1 sensor ON")

    # ────────── save Q-table
    save_q_table_csv(Q, args.output)