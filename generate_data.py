"""
Generate a simulated multi-sensor dataset for the project.
Labels:
  0 = normal
  1 = manipulated
This is synthetic demonstration data, not real-world evidence.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_STEPS = 2400
OUT = Path("data/sensor_data.csv")


def generate_dataset(n_steps=N_STEPS, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    timestamps = pd.date_range("2026-01-01", periods=n_steps, freq="min")

    # Smooth, correlated environmental patterns with measurement noise.
    daily = np.sin(2 * np.pi * t / 1440)
    temperature = 27 + 4 * daily + rng.normal(0, 0.35, n_steps)
    humidity = 65 - 7 * daily + rng.normal(0, 1.2, n_steps)
    soil_moisture = 42 + 2.5 * np.sin(2 * np.pi * t / 720) + rng.normal(0, 0.7, n_steps)

    # Peer sensors are similar but not identical.
    temperature_peer = temperature + rng.normal(0, 0.55, n_steps)
    soil_peer = soil_moisture + rng.normal(0, 1.1, n_steps)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "temperature_peer": temperature_peer,
        "soil_moisture_peer": soil_peer,
    })
    df["label"] = 0
    df["manipulation_type"] = "normal"

    # Inject several types of manipulation into the main sensors.
    # Keep event ranges away from the first rows so lag features are available.
    events = [
        (320, 350, "spike"),
        (650, 710, "gradual_drift"),
        (980, 1040, "frozen"),
        (1330, 1370, "peer_inconsistent"),
        (1710, 1760, "replay"),
        (2050, 2080, "spike"),
    ]

    for start, end, kind in events:
        df.loc[start:end, "label"] = 1
        df.loc[start:end, "manipulation_type"] = kind

        if kind == "spike":
            df.loc[start:end, "soil_moisture"] += 28
        elif kind == "gradual_drift":
            drift = np.linspace(0, 22, end - start + 1)
            df.loc[start:end, "soil_moisture"] += drift
        elif kind == "frozen":
            frozen_value = float(df.loc[start - 1, "soil_moisture"])
            df.loc[start:end, "soil_moisture"] = frozen_value
        elif kind == "peer_inconsistent":
            df.loc[start:end, "temperature"] += 12
        elif kind == "replay":
            replay_start = max(0, start - 180)
            old_values = df.loc[replay_start:replay_start + (end - start), "soil_moisture"].to_numpy()
            if len(old_values) < (end - start + 1):
                old_values = np.resize(old_values, end - start + 1)
            df.loc[start:end, "soil_moisture"] = old_values

    # Ensure a clean, reproducible row order.
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data):,} rows to {OUT}")
    print("Label counts:")
    print(data["label"].value_counts().rename(index={0: "normal", 1: "manipulated"}))
    print("\nManipulation types:")
    print(data["manipulation_type"].value_counts())
