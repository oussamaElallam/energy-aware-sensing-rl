import random
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# Import HealthWearableEnv from framework instead of duplicating code
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import HealthWearableEnv


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Environment (DEPRECATED - use HealthWearableEnv from framework instead)
# ─────────────────────────────────────────────────────────────────────────────
class ThreeSensorTimeEnv_DEPRECATED:
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
    env: HealthWearableEnv,
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
    import pickle
    
    parser = argparse.ArgumentParser(description='Train Q-learning agent for energy-aware sensing')
    parser.add_argument('--lambda_risk', type=float, default=0.0, 
                       help='Risk penalty weight for missed events (default: 0.0)')
    parser.add_argument('--beta', type=float, default=0.008,
                       help='Energy penalty weight (default: 0.008)')
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes (default: 3000)')
    parser.add_argument('--output', type=str, default='q_table_fixed',
                       help='Output Q-table filename base (without extension)')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate convergence plot')
    parser.add_argument('--n_seeds', type=int, default=10,
                       help='Number of different traces to train on (default: 10)')
    args = parser.parse_args()
    
    STEPS_PER_SEED = 12_000  # 16 h at 5 s cadence per seed
    
    # Generate diverse training data from multiple seeds
    # This ensures the Q-table generalizes across different event patterns
    print(f"Generating training data from {args.n_seeds} different seeds...")
    all_scenarios = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        scenario = [
            {
                "arr_flag": int(rng.choice([0, 1], p=[0.7, 0.3])),
                "bp_flag": int(rng.choice([0, 1], p=[0.6, 0.4])),
                "fever_flag": int(rng.choice([0, 1], p=[0.8, 0.2])),
            }
            for _ in range(STEPS_PER_SEED)
        ]
        all_scenarios.append(scenario)
    
    print(f"Total: {args.n_seeds} traces × {STEPS_PER_SEED} steps = {args.n_seeds * STEPS_PER_SEED} training samples")

    # Use HealthWearableEnv from framework for consistency
    # We'll rotate through scenarios each episode
    print(f"\nTraining Q-learning agent with FIXED persistence logic...")
    print(f"Episodes: {args.episodes}, Beta: {args.beta}, Lambda_risk: {args.lambda_risk}")
    print(f"Training on {args.n_seeds} different traces for generalization")
    
    Q: QTable = {}
    rewards: List[float] = []
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.998
    gamma = 0.95
    alpha_lr = 0.1
    
    for ep in range(args.episodes):
        # Rotate through different scenarios each episode
        scenario = all_scenarios[ep % len(all_scenarios)]
        
        env = HealthWearableEnv(
            data=scenario,
            sensor_costs=[10, 4, 1],
            alpha=15.0,
            beta=args.beta,
            lambda_risk=args.lambda_risk,
            max_battery=400_000,
            max_time_steps=STEPS_PER_SEED,
        )
        
        s = env.reset()
        ep_r = 0.0
        done = False
        
        while not done:
            # ε-greedy
            if random.random() < epsilon:
                a = random.randrange(8)
            else:
                a = int(np.argmax([Q.get((s, b), 0.0) for b in range(8)]))
            
            s2, r, done, _ = env.step(a)
            
            best_next = max(Q.get((s2, b), 0.0) for b in range(8))
            td_target = r + gamma * best_next
            td_error = td_target - Q.get((s, a), 0.0)
            Q[(s, a)] = Q.get((s, a), 0.0) + alpha_lr * td_error
            
            s = s2
            ep_r += r
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(ep_r)
        
        if (ep + 1) % 500 == 0:
            print(f"  Episode {ep+1}/{args.episodes}, Avg reward (last 50): {np.mean(rewards[-50:]):.2f}")
    
    R = rewards
    print(f"\nTraining complete! Avg reward (last 50 eps): {np.mean(R[-50:]):.2f}")

    # ────────── PROBE: how many state-actions prefer at least one sensor ON
    on_pref  = sum(1 for (s, a), v in Q.items() if v > 0 and a != 0)
    total_sa = len(Q)
    print(f"{on_pref/total_sa*100:.1f}% of state-actions favour >=1 sensor ON")

    # ────────── save Q-table as CSV
    save_q_table_csv(Q, f"{args.output}.csv")
    
    # ────────── save Q-table as pickle
    pkl_path = Path(f"{args.output}.pkl")
    with pkl_path.open("wb") as f:
        pickle.dump(Q, f)
    print(f"Q-table saved to {pkl_path} (pickle format)")
    
    # ────────── generate convergence plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: Raw rewards
            ax1.plot(R, alpha=0.3, linewidth=0.5)
            # Moving average
            window = 50
            moving_avg = np.convolve(R, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(R)), moving_avg, color='red', linewidth=2, label=f'{window}-ep moving avg')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Training Convergence (Fixed Persistence Logic)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Q-table growth
            ax2.axhline(y=len(Q), color='green', linestyle='--', label=f'Final Q-table size: {len(Q)}')
            ax2.set_xlabel('Training completed')
            ax2.set_ylabel('Q-table entries')
            ax2.set_title('Q-table Size')
            ax2.legend()
            
            plt.tight_layout()
            fig_path = Path(f"{args.output}_convergence.png")
            plt.savefig(fig_path, dpi=150)
            print(f"Convergence plot saved to {fig_path}")
            plt.close()
        except ImportError:
            print("matplotlib not available, skipping plot generation")