from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).with_name("crack_growth_data.csv")


def generate_crack_growth_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic crack growth samples using Paris' Law.

    Paris' Law:
        da / dN = C * (Delta K) ** m

    where:
        a = crack length
        N = load cycles
        C, m = material constants
        Delta K = stress intensity range
    """
    rng = np.random.default_rng(random_state)

    initial_crack_m = rng.uniform(0.0005, 0.012, n_samples)
    stress_range_mpa = rng.uniform(25.0, 180.0, n_samples)
    load_cycles = rng.integers(1_000, 2_000_000, n_samples)

    geometry_factor = rng.normal(1.12, 0.06, n_samples).clip(1.0, 1.35)
    paris_c = rng.lognormal(mean=np.log(2.0e-12), sigma=0.35, size=n_samples)
    paris_m = rng.normal(3.0, 0.18, n_samples).clip(2.5, 3.6)

    delta_k = geometry_factor * stress_range_mpa * np.sqrt(np.pi * initial_crack_m)
    crack_growth_m = paris_c * np.power(delta_k, paris_m) * load_cycles
    crack_length_m = initial_crack_m + crack_growth_m

    fracture_toughness_mpa_sqrt_m = rng.normal(55.0, 8.0, n_samples).clip(30.0, 85.0)
    final_stress_intensity = geometry_factor * stress_range_mpa * np.sqrt(np.pi * crack_length_m)
    critical_crack_m = np.power(
        fracture_toughness_mpa_sqrt_m / (geometry_factor * stress_range_mpa),
        2,
    ) / np.pi

    failure = (
        (crack_length_m >= critical_crack_m)
        | (final_stress_intensity >= fracture_toughness_mpa_sqrt_m)
    ).astype(int)

    return pd.DataFrame(
        {
            "crack_length_mm": crack_length_m * 1000.0,
            "stress_intensity": final_stress_intensity,
            "load_cycles": load_cycles,
            "failure": failure,
        }
    ).round(
        {
            "crack_length_mm": 4,
            "stress_intensity": 4,
        }
    )


def main() -> None:
    data = generate_crack_growth_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data):,} rows to {OUTPUT_PATH}")
    print(data.head())


if __name__ == "__main__":
    main()
