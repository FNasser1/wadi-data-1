
"""
data_generator.py
-----------------
Synthetic petrochemical sensor data generator.

Generates realistic multi-sensor time-series for multiple machines,
including gradual degradation and three failure modes:
- bearing_wear
- imbalance
- misalignment

Features:
- Vibration sensors (X, Y, Z)
- Temperature
- Pressure
- RPM
- Configurable sampling frequency, number of machines, and duration
- Realistic noise, spikes, dropouts (NaNs), and sensor bias drift
- Gradual degradation patterns conditioned on failure mode

Usage (CLI):
    python data_generator.py --output data.csv --days 30 --n_machines 10 --freq "1min" --seed 42

Usage (API):
    from data_generator import generate_dataset
    df, meta = generate_dataset(days=30, n_machines=10, freq="1min", seed=42)
    df.to_csv("data.csv", index=False)

Output columns:
    timestamp, machine_id, rpm, pressure, temperature,
    vib_x, vib_y, vib_z, failure_mode, degradation_level, failure_event

Notes:
- 'degradation_level' ∈ [0, 1] grows after degradation_start_ts for affected machines
- 'failure_event' == 1 near end-of-life window (last hours) for a degraded machine
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import argparse
import math
import random


@dataclass
class GeneratorConfig:
    start: pd.Timestamp
    days: int = 30
    n_machines: int = 10
    freq: str = "1min"
    seed: Optional[int] = 42
    ambient_temp_c: float = 28.0
    ambient_temp_daily_amp: float = 4.0   # daily sinusoid amplitude
    base_pressure_bar: float = 8.0
    pressure_variability: float = 0.25
    rpm_base: Tuple[int, int] = (1450, 1750)  # operating window
    rpm_step_jitter: int = 15
    failure_modes: Tuple[str, ...] = ("bearing_wear", "imbalance", "misalignment")
    frac_degrading: float = 0.6  # fraction of machines that will degrade during period
    dropout_prob: float = 0.0007 # prob per point to drop to NaN (sensor dropout)
    spike_prob: float = 0.0008   # prob per point to insert spike
    bias_drift_ppm_per_day: float = 150.0  # parts-per-million bias drift per day (small)
    failure_window_hours: int = 6  # last window to mark failure_event=1 for degraded machines


def _rng(seed: Optional[int]) -> np.random.Generator:
    s = seed if seed is not None else np.random.SeedSequence().entropy
    return np.random.default_rng(int(s))


def _make_time_index(cfg: GeneratorConfig) -> pd.DatetimeIndex:
    end = cfg.start + pd.Timedelta(days=cfg.days)
    return pd.date_range(cfg.start, end, freq=cfg.freq, inclusive="left")


def _daily_cycle(ts: pd.DatetimeIndex, amplitude: float, phase_hours: float = 15.0) -> np.ndarray:
    # simple daily temperature/pressure cycle
    seconds_in_day = 24*3600
    # peak around 3pm by default (phase)
    phase = phase_hours * 3600
    x = (ts.view('int64')//10**9) % seconds_in_day
    return amplitude * np.sin(2*np.pi * (x - phase) / seconds_in_day)


def _assign_machine_profiles(cfg: GeneratorConfig, rng: np.random.Generator):
    profiles = []
    n_degrading = max(1, int(cfg.n_machines * cfg.frac_degrading))
    degrading_ids = set(rng.choice(np.arange(cfg.n_machines), size=n_degrading, replace=False))
    for m in range(cfg.n_machines):
        mode = None
        deg_start = None
        if m in degrading_ids:
            mode = rng.choice(cfg.failure_modes)
            # degradation starts sometime after day 5
            deg_start_day = rng.integers(low=5, high=max(6, cfg.days-5))
            deg_start = (cfg.start + pd.Timedelta(days=int(deg_start_day)))
        profiles.append({
            "machine_id": f"M{m+1:02d}",
            "failure_mode": mode,  # None means healthy entire period
            "degradation_start": deg_start
        })
    return profiles


def _simulate_rpm(ts: pd.DatetimeIndex, cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    base_low, base_high = cfg.rpm_base
    # random baseline per series
    base = rng.integers(base_low, base_high)
    rpm = np.full(len(ts), base, dtype=float)
    # introduce slow random walks + step changes
    walk = rng.normal(0, 0.5, size=len(ts)).cumsum()
    rpm += walk
    # occasional step changes
    for _ in range(len(ts)//8000 + 1):
        ix = int(rng.integers(0, len(ts)))
        step = rng.normal(0, cfg.rpm_step_jitter)
        rpm[ix:] += step
    # clamp plausible range
    rpm = np.clip(rpm, base_low-100, base_high+100)
    return rpm


def _simulate_pressure(ts: pd.DatetimeIndex, cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    base = cfg.base_pressure_bar + _daily_cycle(ts, amplitude=0.1)
    noise = rng.normal(0, cfg.pressure_variability, size=len(ts))
    p = base + noise
    return np.clip(p, 6.5, 12.0)


def _simulate_temperature(ts: pd.DatetimeIndex, cfg: GeneratorConfig, rng: np.random.Generator, rpm: np.ndarray, deg_level: np.ndarray) -> np.ndarray:
    ambient = cfg.ambient_temp_c + _daily_cycle(ts, amplitude=cfg.ambient_temp_daily_amp)
    # temperature rises with rpm and with degradation (e.g., friction)
    t = ambient + 0.006*(rpm - np.nanmean(rpm)) + 12.0*deg_level
    # add sensor noise
    t += rng.normal(0, 0.6, size=len(ts))
    return np.clip(t, ambient.min()-2, ambient.max()+35)


def _simulate_vibration_axes(
    ts: pd.DatetimeIndex,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    rpm: np.ndarray,
    mode: Optional[str],
    deg_level: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # base vibration levels (mm/s)
    base_x = rng.normal(2.2, 0.35)
    base_y = rng.normal(2.0, 0.35)
    base_z = rng.normal(1.8, 0.35)

    # convert rpm to fundamental frequency (Hz)
    freq = rpm / 60.0
    t_seconds = (ts.view('int64')//10**9).astype(float)
    # construct sinusoids at 1x and 2x shaft frequencies plus noise
    def synth(base, axis_scale=1.0, phase=0.0):
        s1 = 0.6*axis_scale*np.sin(2*np.pi*freq * (t_seconds - t_seconds[0]) + phase)
        s2 = 0.25*axis_scale*np.sin(2*np.pi*2*freq * (t_seconds - t_seconds[0]) + 2*phase)
        return base + s1 + s2

    vx = synth(base_x, axis_scale=1.0, phase=0.0)
    vy = synth(base_y, axis_scale=1.0, phase=0.9)
    vz = synth(base_z, axis_scale=0.8, phase=1.8)

    # apply degradation patterns by failure mode
    if mode == "imbalance":
        # imbalance tends to increase radial vibration (X,Y) at 1x with growing amplitude
        amp_growth = 3.0 * deg_level  # up to +3 mm/s
        vx += amp_growth * np.sin(2*np.pi*freq*(t_seconds - t_seconds[0]))
        vy += 0.8*amp_growth * np.sin(2*np.pi*freq*(t_seconds - t_seconds[0]) + 0.4)
    elif mode == "misalignment":
        # misalignment often increases axial (Z) and introduces higher harmonics
        vz += 2.5*deg_level * np.sin(2*np.pi*2*freq*(t_seconds - t_seconds[0]) + 0.2)
        vx += 0.7*deg_level * np.sin(2*np.pi*3*freq*(t_seconds - t_seconds[0]) + 0.1)
    elif mode == "bearing_wear":
        # bearing wear increases broadband vibration + kurtosis (impulsiveness)
        broadband = rng.normal(0, 0.7 + 2.0*deg_level, size=len(ts))
        vx += broadband * 0.9
        vy += broadband * 0.8
        vz += broadband * 0.7

    # add white noise
    vx += rng.normal(0, 0.25, size=len(ts))
    vy += rng.normal(0, 0.25, size=len(ts))
    vz += rng.normal(0, 0.25, size=len(ts))

    # clamp to plausible ranges
    vx = np.clip(vx, 0.1, 40.0)
    vy = np.clip(vy, 0.1, 40.0)
    vz = np.clip(vz, 0.1, 40.0)
    return vx, vy, vz


def _make_degradation_level(ts: pd.DatetimeIndex, start_ts: Optional[pd.Timestamp], rng: np.random.Generator) -> np.ndarray:
    if start_ts is None:
        return np.zeros(len(ts))
    level = np.zeros(len(ts))
    mask = ts >= start_ts
    if not mask.any():
        return level
    # smooth, accelerating growth after start
    idx = np.where(mask)[0]
    steps = np.linspace(0, 1, num=len(idx))
    curve = steps**1.7  # convex growth
    # add small noise
    curve += rng.normal(0, 0.02, size=len(curve))
    curve = np.clip(curve, 0, 1.0)
    level[idx] = curve
    return level


def _inject_sensor_artifacts(cfg: GeneratorConfig, rng: np.random.Generator, arr_dict: Dict[str, np.ndarray], ts: pd.DatetimeIndex):
    n = len(ts)
    # random NaN dropouts
    for key, arr in arr_dict.items():
        drop_mask = rng.random(n) < cfg.dropout_prob
        arr[drop_mask] = np.nan

    # random spikes
    for key, arr in arr_dict.items():
        spike_mask = rng.random(n) < cfg.spike_prob
        spikes = rng.normal(0, 6.0, size=n) * spike_mask
        arr += spikes

    # small bias drift per day (ppm)
    drift = (np.arange(n) / n) * (cfg.bias_drift_ppm_per_day / 1e6)
    for key, arr in arr_dict.items():
        arr *= (1.0 + drift)


def generate_dataset(
    days: int = 30,
    n_machines: int = 10,
    freq: str = "1min",
    seed: Optional[int] = 42,
    start: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic dataset and metadata.
    Returns:
        df: long-format time-series across machines
        meta: dict with machine profiles and configuration
    """
    if start is None:
        # start on a recent Monday for nicer weekly cycles
        today = pd.Timestamp.utcnow().floor("D")
        start = today - pd.Timedelta(days=(today.weekday()))

    cfg = GeneratorConfig(start=start, days=days, n_machines=n_machines, freq=freq, seed=seed)
    rng = _rng(cfg.seed)
    ts = _make_time_index(cfg)

    profiles = _assign_machine_profiles(cfg, rng)

    frames = []
    for prof in profiles:
        rpm = _simulate_rpm(ts, cfg, rng)

        # degradation profile
        deg_level = _make_degradation_level(ts, prof["degradation_start"], rng)

        temp = _simulate_temperature(ts, cfg, rng, rpm, deg_level)
        press = _simulate_pressure(ts, cfg, rng)

        vx, vy, vz = _simulate_vibration_axes(ts, cfg, rng, rpm, prof["failure_mode"], deg_level)

        # inject artifacts
        arrs = {"rpm": rpm, "pressure": press, "temperature": temp, "vib_x": vx, "vib_y": vy, "vib_z": vz}
        _inject_sensor_artifacts(cfg, rng, arrs, ts)

        df_m = pd.DataFrame({
            "timestamp": ts,
            "machine_id": prof["machine_id"],
            "rpm": arrs["rpm"],
            "pressure": arrs["pressure"],
            "temperature": arrs["temperature"],
            "vib_x": arrs["vib_x"],
            "vib_y": arrs["vib_y"],
            "vib_z": arrs["vib_z"],
            "failure_mode": prof["failure_mode"] if prof["failure_mode"] is not None else "healthy",
            "degradation_level": deg_level,
        })

        # mark failure event in last hours for degraded machines
        if prof["degradation_start"] is not None:
            end_time = ts[-1]
            failure_window_start = end_time - pd.Timedelta(hours=cfg.failure_window_hours)
            df_m["failure_event"] = ((df_m["timestamp"] >= failure_window_start) & (df_m["degradation_level"] > 0.8)).astype(int)
        else:
            df_m["failure_event"] = 0

        frames.append(df_m)

    df = pd.concat(frames, ignore_index=True)
    meta = {"config": cfg.__dict__, "profiles": profiles}
    return df, meta


def main():
    parser = argparse.ArgumentParser(description="Synthetic petrochemical PdM data generator")
    parser.add_argument("--output", type=str, default="synthetic_pdm.csv", help="Output CSV path")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--n_machines", type=int, default=10)
    parser.add_argument("--freq", type=str, default="1min")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, meta = generate_dataset(days=args.days, n_machines=args.n_machines, freq=args.freq, seed=args.seed)
    df.to_csv(args.output, index=False)
    meta_path = args.output.replace(".csv", "_meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, default=str, indent=2)
    print(f"Wrote {args.output} and {meta_path}")


if __name__ == "__main__":
    main()
