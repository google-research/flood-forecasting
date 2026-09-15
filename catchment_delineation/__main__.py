"""Main entry point for running catchment_delineation as a module (python -m catchment_delineation)."""

import sys
from catchment_delineation.cli import main

if __name__ == "__main__":
  sys.exit(main())
