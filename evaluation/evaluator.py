"""
Canonical controller evaluation utilities.

Standardizes RL/PID evaluation so training and dashboard use the same
protocol, energy metric, and telemetry extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from controllers.pid_controller import PIDController
from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator


def _load_config(env_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load config from env_config dictionary or config path."""
    config = env_config.get("config")
    if config is not None:
        return config

    config_path = env_config.get("config_path", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_env(
    config: Dict[str, Any],
    config_path: str,
    pattern_override: Optional[str] = None,
) -> DataCenterThermalEnv:
    """Construct workload generator and thermal environment from config."""
    grid_size = tuple(config["simulation"]["grid_size"])
    pattern = pattern_override or config["workload"]["synthetic_pattern"]

    workload_gen = SyntheticWorkloadGenerator(
        grid_size=grid_size,
        pattern=pattern,
        base_load=config["workload"]["base_load"],
        peak_load=config["workload"]["peak_load"],
    )
    return DataCenterThermalEnv(config_path=config_path, workload_generator=workload_gen)


def _run_episode(
    controller_type: str,
    controller: Any,
    env: DataCenterThermalEnv,
    episode_seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one episode and collect canonical metrics/telemetry."""
    np.random.seed(episode_seed)
    state, _ = env.reset(seed=episode_seed)

    if controller_type == "pid":
        controller.reset()

    temps: List[float] = []
    max_temps: List[float] = []
    coolings: List[float] = []
    rewards: List[float] = []
    violations: List[int] = []

    energy_steps: List[float] = []
    policy_cooling_steps: List[float] = []
    final_cooling_steps: List[float] = []
    override_cooling_steps: List[float] = []

    terminated = False
    truncated = False

    for step in range(max_steps):
        # Deterministic per-step stochasticity for fair RL/PID pairing.
        np.random.seed((episode_seed * 1000003 + step * 9973) % (2**32 - 1))

        if controller_type == "rl":
            action = controller.select_action(state, training=False)
        elif controller_type == "pid":
            grids = env.get_state_grid()
            proposed = controller.compute(grids["temperatures"])
            env.cooling_levels = np.clip(proposed, 0.0, 1.0)
            action = 1
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        state, reward, terminated, truncated, info = env.step(action)
        grids = env.get_state_grid()

        mean_temp = float(np.mean(grids["temperatures"]))
        max_temp = float(np.max(grids["temperatures"]))
        mean_cooling = float(np.mean(grids["cooling_levels"]))

        step_energy = float(np.mean(np.square(grids["cooling_levels"])))

        temps.append(mean_temp)
        max_temps.append(max_temp)
        coolings.append(mean_cooling)
        rewards.append(float(reward))
        violations.append(int(np.sum(grids["temperatures"] > 80.0)))

        energy_steps.append(step_energy)
        policy_cooling_steps.append(float(info.get("policy_cooling", mean_cooling)))
        final_cooling_steps.append(float(info.get("final_cooling", mean_cooling)))
        override_cooling_steps.append(float(info.get("override_cooling", 0.0)))

        if terminated or truncated:
            break

    return {
        "avg_temp": float(np.mean(temps)) if temps else 0.0,
        "max_temp": float(np.max(max_temps)) if max_temps else 0.0,
        "avg_cooling": float(np.mean(coolings)) if coolings else 0.0,
        "avg_energy": float(np.mean(energy_steps)) if energy_steps else 0.0,
        "violations": int(np.sum(violations)),
        "episode_length": len(temps),
        "history": {
            "temps": temps,
            "cooling": coolings,
            "rewards": rewards,
            "violations": violations,
            "energy": energy_steps,
            "energy_cum": list(np.cumsum(energy_steps)),
            "policy_cooling": policy_cooling_steps,
            "final_cooling": final_cooling_steps,
            "override_cooling": override_cooling_steps,
        },
    }


def evaluate_controller(
    controller: Any,
    env_config: Dict[str, Any],
    seed: int,
    episodes: int,
) -> Dict[str, Any]:
    """
    Canonical evaluator required by debugging report.

    Args:
        controller: RL agent or PIDController instance.
        env_config: Dict including controller_type ('rl'|'pid'), optional
            config/config_path/workload_pattern/max_steps.
        seed: Base seed for reproducible episode schedules.
        episodes: Number of episodes.

    Returns:
        Aggregated metrics + per-episode histories.
    """
    config = _load_config(env_config)
    config_path = env_config.get("config_path", "config.yaml")
    controller_type = env_config.get("controller_type", "rl").lower()
    workload_pattern = env_config.get("workload_pattern")
    max_steps = int(env_config.get("max_steps", config["simulation"]["max_steps"]))

    episode_results: List[Dict[str, Any]] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        env = _build_env(config, config_path=config_path, pattern_override=workload_pattern)
        result = _run_episode(
            controller_type=controller_type,
            controller=controller,
            env=env,
            episode_seed=ep_seed,
            max_steps=max_steps,
        )
        episode_results.append(result)

    def _mean(key: str) -> float:
        vals = [r[key] for r in episode_results]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "controller_type": controller_type,
        "episodes": episode_results,
        "avg_temp": _mean("avg_temp"),
        "max_temp": float(np.max([r["max_temp"] for r in episode_results])) if episode_results else 0.0,
        "avg_cooling": _mean("avg_cooling"),
        "avg_energy": _mean("avg_energy"),
        "violations": int(np.sum([r["violations"] for r in episode_results])),
        "avg_episode_length": _mean("episode_length"),
    }


def evaluate_rl_vs_pid(
    rl_agent: Any,
    pid_controller: PIDController,
    env_config: Dict[str, Any],
    seed: int,
    episodes: int,
) -> Dict[str, Any]:
    """Run canonical paired evaluation for RL and PID."""
    rl_result = evaluate_controller(
        controller=rl_agent,
        env_config={**env_config, "controller_type": "rl"},
        seed=seed,
        episodes=episodes,
    )
    pid_result = evaluate_controller(
        controller=pid_controller,
        env_config={**env_config, "controller_type": "pid"},
        seed=seed,
        episodes=episodes,
    )

    pid_energy = pid_result["avg_energy"]
    if pid_energy > 1e-9:
        raw_saved = ((pid_energy - rl_result["avg_energy"]) / pid_energy) * 100.0
    else:
        raw_saved = 0.0

    energy_saved_pct = float(max(min(raw_saved, 100.0), -50.0))

    return {
        "rl": rl_result,
        "pid": pid_result,
        "energy_saved_pct": energy_saved_pct,
    }
