from __future__ import annotations

import argparse
from pathlib import Path

from src.plot_series import plot_overview
from src.simulate_series import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run abTEM single-orientation thickness series simulation")
    parser.add_argument("--config", default="config/simulation.json", help="Path to simulation JSON config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    index_csv = run_simulation(config_path)
    overview_png = plot_overview(index_csv)

    print(f"index_csv={index_csv}")
    print(f"overview_png={overview_png}")


if __name__ == "__main__":
    main()
