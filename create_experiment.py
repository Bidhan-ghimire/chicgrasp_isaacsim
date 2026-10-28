"""Create a named research configuration without starting the SDK."""
from pipeline.setup import create_experiment_main as main

if __name__ == "__main__":
    raise SystemExit(main())
