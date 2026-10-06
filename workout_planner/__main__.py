"""`python -m workout_planner` -> launches the Streamlit app."""

import sys
from pathlib import Path

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
    sys.exit(stcli.main())
