"""
optimizer.py
-------------
Reusable valley-filling (peak-shaving) optimization engine for the
energy management backend.

Given a day's forecasted priority load, controllable load, renewable
supply, and tariff, this decides how much controllable energy to run
in each hour to keep grid draw under a capacity target - shifting
load away from peak hours into hours with spare capacity.

NOTE on relay actuation: this engine outputs a continuous recommended
controllable-load allocation per hour (in kW), since that's the right
level for the optimization math. Your ESP32 relays are binary (on/off)
per zone, so the backend needs one more translation step: mapping this
hourly allocation target down to which of zone1/zone2/zone3 to switch
on in a given hour so their combined power roughly matches the target.
A simple starting rule: at each hour, turn on zones (largest first)
until their summed nominal power reaches the target allocation for
that hour. Refine this once you have real per-zone forecasts running.
"""

import numpy as np
import pandas as pd


def valley_fill(priority: np.ndarray, renewable: np.ndarray, total_energy: float,
                 cap: float, resolution: float = 0.25) -> np.ndarray:
    """Greedily distributes total_energy (kWh) of controllable load across
    24 hours, always adding to whichever hour currently has the most spare
    room under the capacity cap. Returns an array of 24 allocations (kW)."""
    hours = len(priority)
    base = priority - renewable
    alloc = np.zeros(hours)
    remaining = total_energy

    while remaining > 1e-6:
        current_draw = np.clip(base + alloc, 0, None)
        under_cap = current_draw < cap
        candidates = np.where(under_cap)[0] if under_cap.any() else np.arange(hours)
        t = candidates[np.argmin(current_draw[candidates])]
        step = min(resolution, remaining)
        alloc[t] += step
        remaining -= step

    return alloc


def compute_metrics(priority, controllable, renewable, tariff, alloc) -> dict:
    """Computes before/after peak, cost, and renewable-utilization metrics."""
    baseline_draw = np.clip(priority + controllable - renewable, 0, None)
    optimized_draw = np.clip(priority + alloc - renewable, 0, None)

    baseline_cost = float(np.sum(baseline_draw * tariff))
    optimized_cost = float(np.sum(optimized_draw * tariff))
    baseline_peak = float(baseline_draw.max())
    optimized_peak = float(optimized_draw.max())

    baseline_renew_used = float(np.sum(np.minimum(renewable, priority + controllable)))
    optimized_renew_used = float(np.sum(np.minimum(renewable, priority + alloc)))

    return {
        "baseline_peak_kw": round(baseline_peak, 2),
        "optimized_peak_kw": round(optimized_peak, 2),
        "peak_reduction_pct": round((baseline_peak - optimized_peak) / baseline_peak * 100, 1),
        "baseline_cost_rs": round(baseline_cost, 2),
        "optimized_cost_rs": round(optimized_cost, 2),
        "cost_reduction_pct": round((baseline_cost - optimized_cost) / baseline_cost * 100, 1),
        "baseline_renewable_used_kwh": round(baseline_renew_used, 2),
        "optimized_renewable_used_kwh": round(optimized_renew_used, 2),
        "renewable_gain_pct": round((optimized_renew_used - baseline_renew_used) / baseline_renew_used * 100, 1),
        "baseline_grid_draw": baseline_draw.tolist(),
        "optimized_grid_draw": optimized_draw.tolist(),
    }


def optimize_day(priority: np.ndarray, controllable: np.ndarray, renewable: np.ndarray,
                  tariff: np.ndarray, cap_percentile: float = 60) -> dict:
    """Main entry point the backend calls. Takes 24-hour arrays (forecasted or actual)
    for priority load, controllable load, renewable supply, and tariff, and returns
    the recommended hourly controllable allocation plus evaluation metrics.

    cap_percentile controls how aggressive the peak-shaving target is - lower
    values force more shifting (tighter cap), higher values shift less.
    """
    baseline_draw = np.clip(priority + controllable - renewable, 0, None)
    cap = float(np.percentile(baseline_draw, cap_percentile))

    alloc = valley_fill(priority, renewable, float(controllable.sum()), cap)
    metrics = compute_metrics(priority, controllable, renewable, tariff, alloc)
    metrics["capacity_target_kw"] = round(cap, 2)
    metrics["recommended_allocation_kw"] = alloc.round(3).tolist()

    return metrics


if __name__ == "__main__":
    # Smoke test using one day from the synthetic dataset
    df = pd.read_csv('/mnt/user-data/outputs/synthetic_energy_dataset.csv', parse_dates=['Datetime'])
    df = df.set_index('Datetime')
    day = df.loc['2024-06-26']

    priority = day['Zone4_Critical_kW'].values
    controllable = (day['Zone1_Lighting_kW'] + day['Zone2_HVAC_kW'] + day['Zone3_PlugLoads_kW']).values
    renewable = day['Renewable_Supply_kW'].values
    tariff = day['Tariff_Rs_per_kWh'].values

    result = optimize_day(priority, controllable, renewable, tariff)

    print("Optimization result summary:")
    for k, v in result.items():
        if isinstance(v, list):
            continue
        print(f"  {k}: {v}")